#pragma once
#include <type_traits>

namespace util
{
   

template<class T, class FuncA, class FuncB, class Enable = void>
struct Either
{};

// FuncA is member function pointer
template<class T, class FuncA, class FuncB>
struct Either<T, FuncA, FuncB, std::enable_if_t<std::is_member_function_pointer_v<decltype(&FuncA)>>>
{};


void testEither()
{
   struct TestEither {
      void a() {}
      void b() {}
   };

   Either<TestEither, &std::declval<TestEither>.a, &std::declval<TestEither>.b> either;
}
} // namespace util





