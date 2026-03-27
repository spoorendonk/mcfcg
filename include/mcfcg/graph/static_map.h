#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <memory>
#include <ranges>

namespace mcfcg {

template <std::integral K, typename V>
class static_map {
   public:
    using key_type = K;
    using mapped_type = V;
    using size_type = std::size_t;
    using iterator = mapped_type *;
    using const_iterator = const mapped_type *;

   private:
    std::unique_ptr<mapped_type[]> _data;
    size_type _size;

   public:
    constexpr static_map() noexcept : _data(nullptr), _size(0) {}

    constexpr explicit static_map(size_type size)
        : _data(std::make_unique_for_overwrite<mapped_type[]>(size)),
          _size(size) {}

    constexpr static_map(size_type size, const mapped_type & init)
        : static_map(size) {
        std::fill(_data.get(), _data.get() + _size, init);
    }

    template <std::random_access_iterator It>
    constexpr static_map(It begin, It end)
        : static_map(static_cast<size_type>(std::distance(begin, end))) {
        std::copy(begin, end, _data.get());
    }

    template <std::ranges::random_access_range R>
    constexpr explicit static_map(R && r)
        : static_map(std::ranges::begin(r), std::ranges::end(r)) {}

    static_map(const static_map & other)
        : static_map(other.data(), other.data() + other.size()) {}

    constexpr static_map(static_map &&) = default;

    static_map & operator=(const static_map & other) {
        if (_size != other._size) {
            _data = std::make_unique_for_overwrite<mapped_type[]>(other._size);
            _size = other._size;
        }
        std::copy(other.data(), other.data() + other.size(), _data.get());
        return *this;
    }
    static_map & operator=(static_map &&) = default;

    constexpr iterator begin() noexcept { return _data.get(); }
    constexpr iterator end() noexcept { return _data.get() + _size; }
    constexpr const_iterator begin() const noexcept { return _data.get(); }
    constexpr const_iterator end() const noexcept {
        return _data.get() + _size;
    }

    constexpr size_type size() const noexcept { return _size; }

    constexpr mapped_type & operator[](key_type i) noexcept {
        assert(static_cast<size_type>(i) < _size);
        return _data[static_cast<size_type>(i)];
    }
    constexpr const mapped_type & operator[](key_type i) const noexcept {
        assert(static_cast<size_type>(i) < _size);
        return _data[static_cast<size_type>(i)];
    }

    void fill(const mapped_type & v) noexcept {
        std::fill(_data.get(), _data.get() + _size, v);
    }

    constexpr mapped_type * data() noexcept { return _data.get(); }
    constexpr const mapped_type * data() const noexcept { return _data.get(); }
};

}  // namespace mcfcg
