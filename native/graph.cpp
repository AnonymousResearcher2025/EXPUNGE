#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define NO_MANUAL_VECTORIZATION
#include "hnswlib/hnswlib.h"

namespace {
using Id = uint32_t;
constexpr Id none = std::numeric_limits<Id>::max();
thread_local std::string error;
uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
struct Node {
    bool active = false, tomb = false;
    int level = 0;
    std::vector<std::vector<Id>> edges;
};
struct Engine;
struct Space : hnswlib::SpaceInterface<float> {
    Engine *owner;
    explicit Space(Engine *e) : owner(e) {}
    size_t get_data_size() override;
    hnswlib::DISTFUNC<float> get_dist_func() override;
    void *get_dist_func_param() override { return this; }
    static float metric(const void *, const void *, const void *);
};
struct Engine {
    size_t capacity, dim, degree, beam, count = 0, deleted = 0;
    uint64_t seed;
    std::atomic<uint64_t> distances{0}, cache_hits{0};
    bool cosine, hnsw, muted = false;
    float alpha;
    Id entry = none, inserting = none;
    int maxlevel = -1;
    const float *query = nullptr;
    int (*callback)(void *, Id) = nullptr;
    void *context = nullptr;
    std::unique_ptr<Space> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hs;
    std::vector<Node> nodes;
    std::vector<float> payload;
    std::unordered_map<Id, float> warm, scored;
    std::vector<Id> candidates;
    Engine(size_t n, size_t d, bool h, size_t r, size_t l, float a, bool c, uint64_t s)
        : capacity(n), dim(d), degree(r), beam(l), seed(s), cosine(c), hnsw(h), alpha(a), nodes(n) {
        if (!n || !d || r < 2 || l < r || !(a >= 1)) throw std::runtime_error("Invalid graph parameters");
        if (hnsw) {
            space = std::make_unique<Space>(this);
            hs = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), n, r, l, s);
            hs->expunge_read = [this](Id id) { touch(id); };
        } else payload.resize(n * d);
    }
    void touch(Id id) {
        if (id >= capacity) throw std::runtime_error("Graph references an invalid stable ID");
        if (callback && !muted && !callback(context, id)) throw std::runtime_error("Read instrumentation failed");
    }
    float *data(Id id) {
        if (hnsw) return reinterpret_cast<float *>(hs->data_level0_memory_ + id * hs->size_data_per_element_ + hs->offsetData_);
        return payload.data() + id * dim;
    }
    Id pointer_id(const void *p) const {
        if (!hnsw) return none;
        auto addr = reinterpret_cast<uintptr_t>(p);
        auto base = reinterpret_cast<uintptr_t>(hs->data_level0_memory_) + hs->offsetData_;
        if (addr < base) return none;
        auto offset = addr - base;
        if (offset % hs->size_data_per_element_ || offset / hs->size_data_per_element_ >= capacity) return none;
        return Id(offset / hs->size_data_per_element_);
    }
    float distance(const float *a, const float *b, Id cache_id = none) {
        if (cache_id != none && inserting != none) {
            candidates.push_back(cache_id);
            auto it = warm.find(cache_id);
            if (it != warm.end()) { ++cache_hits; scored[cache_id] = it->second; return it->second; }
        }
        ++distances;
        float sum = 0, aa = 0, bb = 0;
        for (size_t j = 0; j < dim; ++j) {
            if (cosine) { sum += a[j] * b[j]; aa += a[j] * a[j]; bb += b[j] * b[j]; }
            else { float delta = a[j] - b[j]; sum += delta * delta; }
        }
        if (!std::isfinite(sum) || !std::isfinite(aa) || !std::isfinite(bb)) throw std::runtime_error("Distance overflow: rescale vectors");
        float result = cosine ? ((aa > 0 && bb > 0) ? float(std::max(0.0, 1.0 - sum / std::sqrt(double(aa) * bb))) : 1.0f) : sum;
        if (cache_id != none && inserting != none) scored[cache_id] = result;
        return result;
    }
    float between(Id a, Id b) {
        touch(a); touch(b);
        Id other = a == inserting ? b : (b == inserting ? a : none);
        return distance(data(a), data(b), other);
    }
    Node get(Id id) {
        if (id >= capacity) throw std::runtime_error("Invalid vertex ID");
        if (!hnsw || !nodes[id].active) return nodes[id];
        Node n = nodes[id];
        n.edges.clear();
        bool previous = muted; muted = true;
        for (int level = 0; level <= n.level; ++level) {
            auto ptr = hs->get_linklist_at_level(id, level);
            size_t size = hs->getListCount(ptr);
            auto ids = reinterpret_cast<Id *>(ptr + 1);
            n.edges.emplace_back(ids, ids + size);
        }
        muted = previous;
        return n;
    }
    void set(Id id, Node n, const float *values) {
        if (id >= capacity) throw std::runtime_error("Invalid vertex ID");
        if (n.active && (n.level < 0 || n.level > 64 || n.edges.size() != size_t(n.level + 1))) throw std::runtime_error("Invalid node levels");
        for (size_t level = 0; level < n.edges.size(); ++level) {
            size_t limit = hnsw && level == 0 ? 2 * degree : degree;
            if (n.edges[level].size() > limit) throw std::runtime_error("Degree exceeds configured bound");
            std::unordered_set<Id> unique;
            for (Id v : n.edges[level]) if (v >= capacity || v == id || !unique.insert(v).second) throw std::runtime_error("Invalid adjacency");
        }
        count += int(n.active) - int(nodes[id].active);
        deleted += int(n.active && n.tomb) - int(nodes[id].active && nodes[id].tomb);
        if (hnsw) {
            if (hs->linkLists_[id]) { free(hs->linkLists_[id]); hs->linkLists_[id] = nullptr; }
            std::memset(hs->data_level0_memory_ + id * hs->size_data_per_element_, 0, hs->size_data_per_element_);
            hs->element_levels_[id] = n.active ? n.level : 0;
            hs->label_lookup_.erase(id);
            if (n.active) {
                hs->label_lookup_[id] = id;
                std::memcpy(hs->getExternalLabeLp(id), &id, sizeof(Id));
                if (n.level) {
                    hs->linkLists_[id] = reinterpret_cast<char *>(calloc(n.level, hs->size_links_per_element_));
                    if (!hs->linkLists_[id]) throw std::bad_alloc();
                }
                bool previous = muted; muted = true;
                for (int level = 0; level <= n.level; ++level) {
                    auto ptr = hs->get_linklist_at_level(id, level);
                    hs->setListCount(ptr, n.edges[level].size());
                    if (!n.edges[level].empty()) std::memcpy(ptr + 1, n.edges[level].data(), n.edges[level].size() * sizeof(Id));
                }
                if (n.tomb) reinterpret_cast<unsigned char *>(hs->get_linklist0(id))[2] |= 1;
                muted = previous;
            }
            hs->cur_element_count = count;
            hs->num_deleted_ = deleted;
        }
        nodes[id] = std::move(n);
        if (values && nodes[id].active) std::memcpy(data(id), values, dim * sizeof(float));
        else if (!nodes[id].active) std::memset(data(id), 0, dim * sizeof(float));
    }
    void metadata(Id e, int l, size_t c, size_t t) {
        if ((e != none && e >= capacity) || c > capacity || t > c) throw std::runtime_error("Invalid global metadata");
        entry = e; maxlevel = l; count = c; deleted = t;
        if (hnsw) { hs->enterpoint_node_ = e; hs->maxlevel_ = l; hs->cur_element_count = c; hs->num_deleted_ = t; }
    }
    std::vector<std::pair<float, Id>> search(const float *q, size_t limit, size_t width, bool include_tomb = false) {
        if (entry == none || !count) return {};
        if (hnsw) {
            auto queue = hs->searchKnnWidth(q, limit, width);
            std::vector<std::pair<float, Id>> out;
            while (!queue.empty()) { out.emplace_back(queue.top().first, Id(queue.top().second)); queue.pop(); }
            std::sort(out.begin(), out.end()); return out;
        }
        width = std::max(width, limit);
        using Pair = std::pair<float, Id>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pending;
        std::priority_queue<Pair> best;
        std::unordered_set<Id> seen;
        auto visit = [&](Id v) {
            touch(v);
            if (!nodes[v].active || !seen.insert(v).second) return;
            float d = distance(q, data(v), query == q ? v : none);
            Pair p(d, v);
            if (best.size() < width || p < best.top()) {
                pending.push(p); best.push(p);
                if (best.size() > width) best.pop();
            }
        };
        visit(entry);
        while (!pending.empty()) {
            auto p = pending.top(); pending.pop();
            if (best.size() == width && p > best.top()) break;
            touch(p.second);
            auto neighbors = nodes[p.second].edges[0];
            for (Id v : neighbors) visit(v);
        }
        std::vector<Pair> out;
        while (!best.empty()) {
            auto p = best.top(); best.pop();
            if (include_tomb || !nodes[p.second].tomb) out.push_back(p);
        }
        std::sort(out.begin(), out.end());
        if (out.size() > limit) out.resize(limit);
        return out;
    }
    std::vector<Id> prune(Id owner, const std::vector<Id>& candidates, size_t limit) {
        std::vector<std::pair<float, Id>> pool;
        std::unordered_set<Id> seen;
        touch(owner);
        for (Id id : candidates) {
            touch(id);
            if (id != owner && nodes[id].active && !nodes[id].tomb && seen.insert(id).second) pool.emplace_back(between(owner, id), id);
        }
        std::sort(pool.begin(), pool.end());
        std::vector<Id> chosen;
        for (auto p : pool) {
            bool occluded = false;
            for (Id picked : chosen) {
                float d = between(picked, p.second);
                if ((hnsw ? d < p.first : alpha * d <= p.first)) { occluded = true; break; }
            }
            if (!occluded) chosen.push_back(p.second);
            if (chosen.size() == limit) break;
        }
        return chosen;
    }
    void insert(Id id, const float *v) {
        if (id >= capacity || nodes[id].active) throw std::runtime_error("Insert requires a fresh stable ID");
        touch(id);
        inserting = id; query = v; scored.clear(); candidates.clear();
        if (hnsw) {
            double uniform = (double(mix(seed ^ id) >> 11) + 0.5) / 9007199254740992.0;
            int level = int(-std::log(uniform) / std::log(double(degree)));
            nodes[id] = Node{true, false, level, {}};
            hs->addPoint(v, id, level, id);
            count = hs->cur_element_count; entry = hs->enterpoint_node_; maxlevel = hs->maxlevel_;
        } else {
            auto candidates = search(v, beam, beam, true);
            Node n{true, false, 0, {{}}};
            set(id, n, v);
            std::vector<Id> ids;
            for (auto p : candidates) ids.push_back(p.second);
            nodes[id].edges[0] = prune(id, ids, degree);
            auto neighbors = nodes[id].edges[0];
            for (Id other : neighbors) {
                touch(other);
                auto next = nodes[other].edges[0]; next.push_back(id);
                nodes[other].edges[0] = next.size() > degree ? prune(other, next, degree) : next;
            }
            if (entry == none) { entry = id; maxlevel = 0; }
        }
        inserting = none; query = nullptr; warm.clear();
    }
};
size_t Space::get_data_size() { return owner->dim * sizeof(float); }
hnswlib::DISTFUNC<float> Space::get_dist_func() { return metric; }
float Space::metric(const void *a, const void *b, const void *p) {
    auto e = static_cast<const Space *>(p)->owner;
    Id id = a == e->query ? e->pointer_id(b) : (b == e->query ? e->pointer_id(a) : none);
    return e->distance(static_cast<const float *>(a), static_cast<const float *>(b), id);
}
template<class F> int guarded(F f) { try { f(); return 0; } catch (const std::exception& e) { error = e.what(); return -1; } }
}
extern "C" {
const char *ex_error() { return error.c_str(); }
void *ex_create(uint32_t n, uint32_t d, int h, uint32_t r, uint32_t l, float a, int cosine, uint64_t seed) {
    try { return new Engine(n, d, h, r, l, a, cosine, seed); } catch (const std::exception& e) { error = e.what(); return nullptr; }
}
void ex_destroy(void *p) { delete static_cast<Engine *>(p); }
void ex_callback(void *p, int (*fn)(void *, Id), void *ctx) { auto e = static_cast<Engine *>(p); e->callback = fn; e->context = ctx; }
int ex_insert(void *p, Id id, const float *v) { return guarded([&] { static_cast<Engine *>(p)->insert(id, v); }); }
int ex_get(void *p, Id id, uint32_t *out, size_t capacity) {
    int length = -1;
    int status = guarded([&] {
        Node n = static_cast<Engine *>(p)->get(id);
        size_t size = 3;
        for (auto& layer : n.edges) size += 1 + layer.size();
        length = int(size);
        if (!out) return;
        if (capacity < size) throw std::runtime_error("Snapshot buffer too small");
        *out++ = n.active; *out++ = n.tomb; *out++ = n.level;
        for (auto& layer : n.edges) { *out++ = layer.size(); for (Id v : layer) *out++ = v; }
    });
    return status ? -1 : length;
}
int ex_set(void *p, Id id, const uint32_t *raw, size_t size, const float *v) {
    return guarded([&] {
        if (size < 3) throw std::runtime_error("Truncated node");
        Node n{bool(raw[0]), bool(raw[1]), int(raw[2]), {}};
        size_t cursor = 3;
        if (n.active) for (int l = 0; l <= n.level; ++l) {
            if (cursor >= size || raw[cursor] > size - cursor - 1) throw std::runtime_error("Truncated adjacency");
            size_t count = raw[cursor++]; n.edges.emplace_back(raw + cursor, raw + cursor + count); cursor += count;
        }
        if (cursor != size) throw std::runtime_error("Trailing node data");
        static_cast<Engine *>(p)->set(id, std::move(n), v);
    });
}
void ex_global(void *p, uint64_t *out) {
    auto e = static_cast<Engine *>(p);
    out[0] = e->entry; out[1] = uint64_t(int64_t(e->maxlevel)); out[2] = e->count; out[3] = e->deleted;
}
int ex_metadata(void *p, uint32_t entry, int level, uint64_t count, uint64_t deleted) { return guarded([&] { static_cast<Engine *>(p)->metadata(entry, level, count, deleted); }); }
int ex_query(void *p, const float *q, uint32_t k, uint32_t width, uint32_t *ids, float *distances) {
    int length = -1;
    int status = guarded([&] {
        auto out = static_cast<Engine *>(p)->search(q, k, width);
        length = int(out.size());
        for (size_t j = 0; j < out.size(); ++j) { distances[j] = out[j].first; ids[j] = out[j].second; }
    });
    return status ? -1 : length;
}
int ex_prune(void *p, Id owner, const Id *candidates, size_t size, size_t limit, Id *out) {
    int length = -1;
    int status = guarded([&] { auto ids = static_cast<Engine *>(p)->prune(owner, std::vector<Id>(candidates, candidates + size), limit); length = ids.size(); std::copy(ids.begin(), ids.end(), out); });
    return status ? -1 : length;
}
void ex_warm(void *p, Id id, float distance) { static_cast<Engine *>(p)->warm[id] = distance; }
int ex_scored(void *p, Id *ids, float *distances) {
    auto& map = static_cast<Engine *>(p)->scored;
    int j = 0;
    for (auto pair : map) { if (ids) ids[j] = pair.first; if (distances) distances[j] = pair.second; ++j; }
    return j;
}
int ex_candidates(void *p, Id *out) { auto& ids = static_cast<Engine *>(p)->candidates; if (out) std::copy(ids.begin(), ids.end(), out); return int(ids.size()); }
void ex_counters(void *p, uint64_t *out) { auto e = static_cast<Engine *>(p); out[0] = e->distances.load(); out[1] = e->cache_hits.load(); }
}
