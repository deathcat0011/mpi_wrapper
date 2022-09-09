#pragma once

#include <cassert>
#include <numeric>
#include <type_traits>
#include <array>

#include <mpi.h>

namespace mpi
{
#  undef WRAPDECL
#  define WRAPDECL static inline constexpr
   using Communicator = MPI_Comm;

   //? NOTICE not going to fix, because we only support contiguous data
   template<typename T>
   using innerType = std::remove_cv_t<
      std::remove_reference_t<
         std::remove_pointer_t <decltype(((T*)nullptr)->operator[](0))>
      >
   >;


#  pragma region FUNDAMENTAL MPI TYPES

   namespace types
   {
      template <typename T>
      struct FundamentalType : public std::false_type
      {};

      template<>
      struct FundamentalType<char> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_CHAR;
         }
      };

      template<>
      struct FundamentalType<unsigned char> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_UNSIGNED_CHAR;
         }
      };

      template<>
      struct FundamentalType<signed char> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_SIGNED_CHAR;
         }
      };

      template<>
      struct FundamentalType<short> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_SHORT;
         }
      };

      template<>
      struct FundamentalType<unsigned short> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_UNSIGNED_SHORT;
         }
      };

      template<>
      struct FundamentalType<int> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_INT;
         }
      };
      
      template<>
      struct FundamentalType<unsigned> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_UNSIGNED;
         }
      };

      template<>
      struct FundamentalType<long> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_LONG;
         }
      };

      template<>
      struct FundamentalType<long long> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_LONG_LONG;
         }
      };

      template<>
      struct FundamentalType<unsigned long> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_UNSIGNED_LONG;
         }
      };

      template<>
      struct FundamentalType<unsigned long long> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_UNSIGNED_LONG_LONG;
         }
      };

      template<>
      struct FundamentalType<float> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_FLOAT;
         }
      };

      template<>
      struct FundamentalType<double> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_DOUBLE;
         }
      };

      template<>
      struct FundamentalType<long double> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_LONG_DOUBLE;
         }
      };
#if 0
#     if defined(MPI_CXX_FLOAT_COMPLEX) || defined(MPI_CXX_DOUBLE_COMPLEX) || defined(MPI_CXX_LONG_DOUBLE_COMPLEX)
#     include <complex>
#     endif

#     ifdef MPI_CXX_FLOAT_COMPLEX
      template<>
      struct FundamentalType< std::complex<float>> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_CXX_FLOAT_COMPLEX;
         }
      };
#     endif

#     ifdef MPI_CXX_DOUBLE_COMPLEX
      template<>
      struct FundamentalType<std::complex<double>> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_CXX_DOUBLE_COMPLEX;
         }
      };
#     endif

#     ifdef MPI_CXX_LONG_DOUBLE_COMPLEX
      template<>
      struct FundamentalType<std::complex<long double>> : public std::true_type
      {
         WRAPDECL MPI_Datatype get()
         {
            return MPI_CXX_LONG_DOUBLE_COMPLEX;
         }
      };
#     endif
#endif


      template<typename T, typename = ::std::enable_if<FundamentalType<T>::value>>
      inline constexpr MPI_Datatype get()
      {
         return (MPI_Datatype) FundamentalType<T>::get();
      }

   } // namespace types

#  pragma endregion FUNDAMENTAL MPI TYPES

// WRAPPER FOR DATA MEMBER FUNCTION ACCESS
#  pragma region DATA
   template<typename T, typename Enable = void>
   struct hasDataMut : public std::false_type
   {};

   template<typename T>
   struct hasDataMut <T, std::true_type> : public std::true_type
   {
      using T_ = std::remove_cv_t<T>;
      static_assert(std::is_same_v<T_*, decltype(((T*)nullptr)->data())>, "T::data must return mutable type ptr" );
   };

   template<typename T, typename Enable = void>
   struct hasDataConst : public std::false_type
   {};

   template<typename T>
   struct hasDataConst <T, std::true_type> : public std::true_type
   {
      using T_ = std::remove_cv_t<T>;
      static_assert(std::is_same_v<const T_*, decltype(((T*)nullptr)->data())>, "T::data must return const type ptr" );
   };

   template<typename T>
   struct hasData : public hasDataMut<T>, public hasDataConst<T>
   {};

   template<typename T>
   struct Data : public hasData<T>
   {
      typedef decltype(((T*)nullptr)->data()) return_type;
      WRAPDECL return_type data(T& t)
      {
         return t.data();
      }
   };

   template<typename T>
   struct ConstData : public hasDataConst<T>
   {
      typedef decltype(((T*)nullptr)->data()) return_type;
      WRAPDECL return_type const data(const T& t)
      {
         return (return_type const ) t.data();
      }
   };

#  pragma endregion DATA

// WRAPPER FOR CONTAINER TO DETERMINE SIZE FOR EITHER size() or length() MEMBER FUNCTIONS
#  pragma region SIZE
   template<typename T, typename Enable = void>
   struct Size{};

   template<class T>
   struct Size<T, typename std::enable_if<std::is_member_function_pointer_v<decltype(&T::size)>>::type>
   {
      typedef decltype(((T*)nullptr)->size()) return_type;
      WRAPDECL return_type size(const T& t)
      {
         return t.size();
      }
   };

   template<class T>
   struct Size<T, typename std::enable_if<std::is_member_function_pointer_v<decltype(&T::length)>>::type >
   {
      typedef decltype(((T*)nullptr)->length()) return_type;
      WRAPDECL return_type size(const T& t)
      {
         return t.length();
      }
   };
# pragma endregion SIZE

// WRAPPER FOR MULTI DIM ARRAY INCREMENT, DEFAULTS TO 1 FOR 1D ARRAYS (WITHOUT FIELD INC)
#  pragma region INC
   template<typename T, typename Enable = void>
   struct Inc{
      WRAPDECL size_t inc(const T& t)
      {
         return 1;
      }
   };

   template<class T>
   struct Inc<T, typename std::enable_if<std::is_member_function_pointer_v<decltype(&T::inc)>>::type>
   {
      typedef decltype(((T*)nullptr)->inc()) return_type;
      WRAPDECL return_type inc(const T& t)
      {
         return t.inc();
      }
   };

# pragma endregion INC

// WRAPPER TO SET CONTAINER CONTENTS
#  pragma region SETTER
   template<typename Source, typename Dest, typename AllowsDirectInit = void>
   struct Setter{};

   
   template<typename Source, typename Dest>
   struct Setter<Source, Dest, std::true_type>
   {
      WRAPDECL Dest wrap(Source&& src)
      {
         return Dest(src.data());
      }

      WRAPDECL Dest wrap(const Source& src)
      {
         return Dest(src.data());
      }

   };

   template<typename Source, typename Dest>
   struct Setter<Source, Dest, std::false_type>
   {
      WRAPDECL Dest wrap(const Source& src)
      {
         Dest ret;
         std::copy( std::begin(src), std::end(src), std::begin(ret));
         return ret;
      }

      WRAPDECL Dest wrap(Source&& src)
      {
         Dest ret;
         std::move(std::begin(src), std::end(src), std::begin(ret));
         return ret;
      }
   };
#  pragma endregion SETTER

   inline int commSize(Communicator communicator = MPI_COMM_WORLD)
   {
      int ret;
      MPI_Comm_size(communicator, &ret);
      return ret;
   }

   inline int commRank(Communicator communicator = MPI_COMM_WORLD)
   {
      int ret;
      MPI_Comm_rank(communicator, &ret);
      return ret;
   }

// GENERATE NEW MPI DATATYPE FOR CONTAINER BASED ON STORED VALUE TYPE
   template <typename T>
   MPI_Datatype get_type(const T& t)
   {
      auto oldtype = types::get<innerType<T>>();

      MPI_Datatype datatype;
      int err = MPI_Type_vector( Size<T>::size(t), 1, Inc<T>::inc(t), oldtype, &datatype );
      assert(err == 0);
      return datatype;
   }

// RAII WRAPPER FOR MPI_DATATYPE
   template<typename T>
   class Datatype
   {

   private:
      int m_count;
      MPI_Datatype m_type;
   public:
      Datatype(const T& t) :
         m_count(Size<T>::size(t)),
         m_type(get_type<T>(t))
      {
         MPI_Type_commit(&m_type);
      }
      ~Datatype()
      {
         // MPI_Type_free(&m_type);
      }

      int count() const {return m_count;}
      MPI_Datatype type() {return m_type;}

      operator MPI_Datatype() const {return m_type;}
   };

   class Context
   {
   private:
      unsigned m_proc;
      unsigned m_rank;
      Communicator m_communicator;
   public:
      Context(int *argc_p, char ***argv_p, Communicator communicator = MPI_COMM_WORLD) :
         m_communicator(communicator)
      {
         MPI_Init(argc_p, argv_p);
         m_proc = ::mpi::commSize(communicator);
         m_rank = ::mpi::commRank(communicator);
      }

      ~Context()
      {
         MPI_Finalize();
      }

      int proc() { return m_proc; }
      int rank() { return m_rank; }

      template<typename T>
      void send(const Datatype<T>& dtype, const T& buffer, int to, int tag = 0) const
      {
         auto count = Size<T>::size(buffer);
         int err =  MPI_Send( ConstData<T>::data(buffer), 1, (MPI_Datatype) dtype, to, tag, m_communicator);
         assert(err == 0);
      }

      template<typename T>
      void recv(const Datatype<T>& dtype, T& buffer, int from, int tag = 0, const size_t q_length = 512) const
      {

         MPI_Status status;
         MPI_Recv(Data<T>::data(buffer), q_length, (MPI_Datatype) dtype, from, tag, m_communicator, &status);
         int num;
         MPI_Get_count(&status, dtype, &num);
         assert(num == 1);
      }
   };

}

