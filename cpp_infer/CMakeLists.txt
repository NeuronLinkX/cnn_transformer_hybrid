# 최신 LibTorch 패키지를 다루기에 충분한 CMake 버전을 요구한다.
cmake_minimum_required(VERSION 3.18)

# 과제 맥락에 맞게 추론 프로젝트 이름을 지정한다.
project(cifar10_hybrid_infer LANGUAGES CXX)

# C++17 표준으로 빌드한다.
set(CMAKE_CXX_STANDARD 17)
# 요청한 C++ 표준을 엄격하게 강제한다.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 컴파일러 전용 언어 확장을 비활성화한다.
set(CMAKE_CXX_EXTENSIONS OFF)

# CMAKE_PREFIX_PATH로부터 LibTorch 패키지를 찾는다.
find_package(Torch REQUIRED)
# 이 실행 파일에 실제로 필요한 OpenCV 모듈만 찾는다.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# LibTorch 컴파일 플래그를 이 타깃에 반영한다.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Homebrew PyTorch가 함께 설치된 macOS 시스템에서는 TorchConfig가 외부 libc10을
# 먼저 발견할 수 있다. 실행 시점에 서로 다른 OpenMP/Torch 스택이 섞이지 않도록
# 명시적 libc10 항목을 현재 선택된 Torch 패키지의 TORCH_INSTALL_PREFIX로 맞춘다.
if(APPLE AND DEFINED TORCH_INSTALL_PREFIX)
  # 이전에 발견된 명시적 libc10 경로를 제거한다.
  list(FILTER TORCH_LIBRARIES EXCLUDE REGEX ".*/libc10\\.dylib$")
  # 현재 활성 Torch 패키지 prefix의 libc10만 다시 추가한다.
  list(APPEND TORCH_LIBRARIES "${TORCH_INSTALL_PREFIX}/lib/libc10.dylib")
endif()

# 단일 이미지 추론 실행 파일을 빌드한다.
add_executable(cifar10_infer main.cpp)
# 실행 파일에 LibTorch와 OpenCV를 링크한다.
target_link_libraries(cifar10_infer PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# OpenCV include 경로를 컴파일러에 노출한다.
target_include_directories(cifar10_infer PRIVATE ${OpenCV_INCLUDE_DIRS})

# MSVC 빌드에서는 LibTorch DLL을 실행 파일 옆으로 복사한다.
if(MSVC)
  # LibTorch 런타임 DLL 목록을 수집한다.
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  # 실행 파일 빌드 후 런타임 DLL을 복사한다.
  add_custom_command(TARGET cifar10_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:cifar10_infer>)
endif()
