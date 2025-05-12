#pragma once

#include <cstring>
#include <iostream>
#include <ostream>
#include <type_traits>
#include <utility>

namespace std {
// clang-format off
template<class T> struct remove_ref_or_p { using type = T; };
template<class T> struct remove_ref_or_p<T*> { using type = T; };
template<class T> struct remove_ref_or_p<T* const> { using type = T; };
template<class T> struct remove_ref_or_p<T* volatile> { using type = T; };
template<class T> struct remove_ref_or_p<T* const volatile> { using type = T; };

template<class T> struct remove_ref_or_p<T*&> { using type = T*; };
template<class T> struct remove_ref_or_p<T* const &> { using type = T* const; };
template<class T> struct remove_ref_or_p<T* volatile &> { using type = T* const volatile; };
template<class T> struct remove_ref_or_p<T* const volatile &> { using type = T* const volatile; };
// clang-format on

template <class T>
using remove_ref_or_p_t = remove_ref_or_p<T>::type;

template <class T>
using remove_ref_and_p_t = remove_pointer_t<remove_reference_t<T>>;

// clang-format off
template <bool b, class T> struct add_pointer_if { using type = T; };
template <class T> struct add_pointer_if<true, T> { using type = std::add_pointer_t<T>; };
// clang-format on

};	   // namespace std

template <typename T>
struct AutoRef {
	// a reference if possible or a owned object otherwise
	T ref;

	template <class Q>
	AutoRef(Q &value) : ref(value) {}
	template <class Q>
	AutoRef(Q &&value) : ref(std::move(value)) {}

	// removing the reference is needed for cases where T is a reference itself
	std::remove_reference_t<T>		 *operator->() { return &ref; }
	const std::remove_reference_t<T> *operator->() const { return &ref; }

	// reference to reference is illegal and thus T& is always deduced as just
	// a reference even when T is a reference type
	T		&operator*() { return ref; }
	const T &operator*() const { return ref; }

	// allow casting to mimicked type
	operator T &() { return ref; }
	operator const T &() const { return ref; }

	T		&get() { return ref; }
	const T &get() const { return ref; }
};

// template deduction guide so that T can be deduced to Q ot Q&
template <class Q>
AutoRef(Q &value) -> AutoRef<Q &>;

template <class Q>
AutoRef(Q &&value) -> AutoRef<Q>;

template <class T>
class SmartAuto {
   public:
	using Internal = std::remove_ref_and_p_t<T>;
	using type	   = T;

	template <class Q>
	SmartAuto(Q *value) : val(value) {}

	template <class Q>
		requires(!std::is_pointer_v<Q>)
	SmartAuto(Q &value) : val(&value) {}

	~SmartAuto()
		requires std::is_pointer_v<std::remove_reference_t<T>>
	{
		delete val;
	}
	~SmartAuto()
		requires(!std::is_pointer_v<std::remove_reference_t<T>>)
	{}

	SmartAuto &operator=(const T &val)
		requires(!std::is_pointer_v<T>)
	{
		this->val = &val;
		return *this;
	}
	SmartAuto &operator=(std::remove_reference_t<T> &&val)
		requires(!std::is_pointer_v<T>)
	{
		*this->val = val;
		return *this;
	}
	SmartAuto &operator=(SmartAuto &&rhs)
		requires std::is_pointer_v<T>
	{
		std::swap(val, rhs.val);
		return *this;
	}
	SmartAuto &operator=(const SmartAuto &rhs)
		requires std::is_pointer_v<T>
	= delete;
	SmartAuto &operator=(const SmartAuto &rhs)
		requires(!std::is_pointer_v<T>)
	{
		this->val = rhs.val;
		return *this;
	}

	operator T &()
		requires(!std::is_pointer_v<T>)
	{
		return *val;
	}
	operator T()
		requires std::is_pointer_v<T>
	{
		return val;
	}

	T get()
		requires std::is_pointer_v<T>
	{
		return val;
	}
	T get()
		requires(!std::is_pointer_v<T>)
	{
		return *val;
	}
	const T get() const
		requires std::is_pointer_v<T>
	{
		return val;
	}
	const T get() const
		requires(!std::is_pointer_v<T>)
	{
		return *val;
	}

	Internal	   *operator->() { return val; }
	const Internal *operator->() const { return val; }
	T			   &operator*() { return *val; }
	const T		   &operator*() const { return *val; }

   private:
	Internal *val;
};

template <class Q>
SmartAuto(Q *value) -> SmartAuto<Q *>;
template <class Q>
SmartAuto(Q *&value) -> SmartAuto<Q *>;

template <class Q>
SmartAuto(Q &value) -> SmartAuto<Q &>;

class Cloneable {
	virtual Cloneable *clone() const = 0;
};

template <class T>
class SmartRef {
	SmartRef(T *ref, bool isRef) : ref(ref), isRef(isRef) {}

   public:
	SmartRef(T &ref) : ref(&ref), isRef(true) {}
	SmartRef(T *ref) : ref(ref), isRef(false) {}
	SmartRef(const SmartRef &ref) : ref(ref.ref), isRef(true) {}
	SmartRef(SmartRef &&ref) noexcept : isRef(true) {
		this->ref = ref.ref;
		std::swap(isRef, ref.isRef);
	}
	SmartRef(T &&ref)
		requires(!std::is_abstract_v<T>)
		: ref(new T(std::move(ref))), isRef(false) {}

	SmartRef(T &&ref)
		requires(std::is_abstract_v<T> && std::is_base_of_v<Cloneable, T>)
		: ref(ref.clone()), isRef(false) {}

	~SmartRef() {
		if (!isRef) { delete ref; }
	}

	T		&operator*() { return *ref; }
	const T &operator*() const { return *ref; }
	T		*operator->() { return ref; }
	const T *operator->() const { return ref; }
	T		*operator&() { return ref; }
	const T *operator&() const { return ref; }
	explicit operator T *() { return ref; }
	explicit operator const T *() const { return ref; }
	explicit operator T &() { return *ref; }
	explicit operator const T &() const { return *ref; }
	template <class U>
	operator SmartRef<U>() && {
		auto ret = SmartRef<U>((U *)ref, isRef);
		isRef	 = true;
		return ret;
	}

	bool operator==(const SmartRef<T> &other) const { return *ref == *other.ref; }

	SmartRef &operator=(T &&ref) {
		*this->ref = std::move(ref);
		return *this;
	}

	SmartRef &operator=(T &ref) {
		*this->ref = ref;
		return *this;
	}

	SmartRef &operator=(T *ref) {
		if (!isRef) { delete this->ref; }
		this->ref = ref;
		isRef	  = false;
		return *this;
	}

	SmartRef &operator=(SmartRef &ref) {
		if (&ref == this->ref) return *this;
		if (!isRef) { delete this->ref; }
		this->ref = ref.ref;
		isRef	  = true;
		return *this;
	}

	SmartRef &operator=(SmartRef &&ref) {
		if (&ref == this->ref) return *this;
		if (!isRef) { delete this->ref; }
		this->ref = ref.ref;
		isRef	  = ref.isRef;
		ref.isRef = true;
		return *this;
	}

	bool isRef;

	template <class U>
	friend class SmartRef;

   private:
	T *ref;
};

template <class T>
class SmartRefArr {
private:
	//SmartRefArr(T *ref, bool isRef) : ref(ref), isRef(isRef) {}

   public:
	SmartRefArr(T &ref) : ref(&ref), isRef(true) {}
	SmartRefArr(T *ref) : ref(ref), isRef(false) {}
	SmartRefArr(const SmartRefArr &ref) : ref(ref.ref), isRef(true) {}
	SmartRefArr(SmartRefArr &&ref) noexcept : isRef(true) {
		this->ref = ref.ref;
		std::swap(isRef, ref.isRef);
	}

	SmartRefArr(const T *ref, std::size_t size) : ref(new T[size]), isRef(false) {
		for (std::size_t i = 0; i < size; ++i) {
			this->ref[i] = ref[i];
		}
	}
	//	SmartRefArr(T &&ref)
	//		requires(!std::is_abstract_v<T>)
	//		: ref(new T(std::move(ref))), isRef(false) {}
	//
	//	SmartRefArr(T &&ref)
	//		requires(std::is_abstract_v<T> && std::is_base_of_v<Cloneable, T>)
	//		: ref(ref.clone()), isRef(false) {}

	~SmartRefArr() {
		if (!isRef) { delete[] ref; }
	}

	T		&operator*() { return *ref; }
	const T &operator*() const { return *ref; }
	T		*operator->() { return ref; }
	const T *operator->() const { return ref; }
	T		*operator&() { return ref; }
	const T *operator&() const { return ref; }

	T		&operator[](std::size_t index) { return ref[index]; }
	const T &operator[](std::size_t index) const { return ref[index]; }

	explicit operator T *() { return ref; }
	explicit operator const T *() const { return ref; }
	explicit operator T &() { return *ref; }
	explicit operator const T &() const { return *ref; }
	template <class U>
	operator SmartRefArr<U>() && {
		auto ret = SmartRefArr<U>((U *)ref, isRef);
		isRef	 = true;
		return ret;
	}

	bool operator==(const SmartRefArr<T> &other) const { return *ref == *other.ref; }

	SmartRefArr &operator=(T &&ref) {
		*this->ref = std::move(ref);
		return *this;
	}

	SmartRefArr &operator=(T &ref) {
		*this->ref = ref;
		return *this;
	}

	SmartRefArr &operator=(T *ref) {
		if (!isRef) { delete [] this->ref; }
		this->ref = ref;
		isRef	  = false;
		return *this;
	}

	SmartRefArr &operator=(SmartRefArr &ref) {
		if (&ref == this->ref) return *this;
		if (!isRef) { delete [] this->ref; }
		this->ref = ref.ref;
		isRef	  = true;
		return *this;
	}

	SmartRefArr &operator=(SmartRefArr &&ref) {
		if (&ref == this->ref) return *this;
		if (!isRef) { delete [] this->ref; }
		this->ref = ref.ref;
		isRef	  = ref.isRef;
		ref.isRef = true;
		return *this;
	}

	template <class U>
	friend class SmartRefArr;

   private:
	T *ref;

   public:
	bool isRef;
};
