#ifndef STOV_INCLUDED
#define STOV_INCLUDED
#pragma warning(push, 0)
#include <string>
#include <type_traits>
#pragma warning(pop)

namespace
{
	template<typename T, typename... ARGS>
	inline std::enable_if_t<std::is_same<std::remove_const_t<std::remove_reference_t<T>>, int>::value, int> stov(ARGS&&... args)
	{
		return std::stoi(std::forward<ARGS>(args)...);
	}

	template<typename T, typename... ARGS>
	inline std::enable_if_t<std::is_same<std::remove_const_t<std::remove_reference_t<T>>, long>::value, long> stov(ARGS&&... args)
	{
		return std::stol(std::forward<ARGS>(args)...);
	}

	template<typename T, typename... ARGS>
	inline std::enable_if_t<std::is_same<std::remove_const_t<std::remove_reference_t<T>>, long long>::value, long long> stov(ARGS&&... args)
	{
		return std::stoll(std::forward<ARGS>(args)...);
	}

	template<typename T, typename... ARGS>
	inline std::enable_if_t<std::is_same<std::remove_const_t<std::remove_reference_t<T>>, unsigned long>::value, unsigned long> stov(ARGS&&... args)
	{
		return std::stoul(std::forward<ARGS>(args)...);
	}

	template<typename T, typename... ARGS>
	inline std::enable_if_t<std::is_same<std::remove_const_t<std::remove_reference_t<T>>, unsigned long long>::value, unsigned long long> stov(ARGS&&... args)
	{
		return std::stoull(std::forward<ARGS>(args)...);
	}

	template<typename T, typename... ARGS>
	inline std::enable_if_t<std::is_same<std::remove_const_t<std::remove_reference_t<T>>, float>::value, float> stov(ARGS&&... args)
	{
		return std::stof(std::forward<ARGS>(args)...);
	}

	template<typename T, typename... ARGS>
	inline std::enable_if_t<std::is_same<std::remove_const_t<std::remove_reference_t<T>>, double>::value, double> stov(ARGS&&... args)
	{
		return std::stod(std::forward<ARGS>(args)...);
	}

	template<typename T, typename... ARGS>
	inline std::enable_if_t<std::is_same<std::remove_const_t<std::remove_reference_t<T>>, long double>::value, long double> stov(ARGS&&... args)
	{
		return std::stold(std::forward<ARGS>(args)...);
	}
}

#endif

