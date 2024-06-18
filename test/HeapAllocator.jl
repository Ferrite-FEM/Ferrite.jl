using Test, Ferrite.HeapAllocator

@testset "HeapAllocator.jl" begin

    # SizedPtr{T}
    sz = 1024 % UInt
    mptr = Ptr{UInt8}(Libc.malloc(sz))
    ptr = HeapAllocator.SizedPtr(mptr, sz)
    @test Ptr{UInt8}(ptr) === ptr.ptr === mptr
    @test Ptr{Int}(ptr) === Ptr{Int}(ptr.ptr) === Ptr{Int}(mptr)
    @test ptr.size == sz

    # Basic malloc, realloc, free
    heap = HeapAllocator.Heap()
    ptr = HeapAllocator.malloc(heap, 1024)
    @test ptr isa HeapAllocator.SizedPtr{UInt8}
    ptr′ = HeapAllocator.realloc(heap, ptr, 2048)
    @test ptr′ isa HeapAllocator.SizedPtr{UInt8}
    @test_throws ErrorException("free: double free") HeapAllocator.free(heap, ptr)
    HeapAllocator.free(heap, ptr′)
    @test_throws ErrorException("free: double free") HeapAllocator.free(heap, ptr′)

    # Internal page allocation: exhaust some pages
    heap = HeapAllocator.Heap()
    ptrs = HeapAllocator.SizedPtr{UInt8}[]
    for _ in 1:(HeapAllocator.MALLOC_PAGE_SIZE ÷ 512 * 2 + 1)
        ptr = HeapAllocator.malloc(heap, 512)
        push!(ptrs, ptr)
    end
    @test length(heap.size_heaps[10].pages) == 3
    @test all(!, heap.size_heaps[10].pages[1].freelist)
    @test all(!, heap.size_heaps[10].pages[2].freelist)
    @test !heap.size_heaps[10].pages[3].freelist[1]
    @test all(heap.size_heaps[10].pages[3].freelist[2:end])
    ptrs′ = HeapAllocator.SizedPtr{UInt8}[]
    for ptr in ptrs
        ptr′ = HeapAllocator.realloc(heap, ptr, 1024)
        @test_throws ErrorException("free: double free") HeapAllocator.free(heap, ptr)
        push!(ptrs′, ptr′)
    end
    @test length(heap.size_heaps[10].pages) == 3 # TODO
    @test all(heap.size_heaps[10].pages[1].freelist)
    @test all(heap.size_heaps[10].pages[2].freelist)
    @test all(heap.size_heaps[10].pages[3].freelist)
    @test length(heap.size_heaps[11].pages) == 5
    @test all(!, heap.size_heaps[11].pages[1].freelist)
    @test all(!, heap.size_heaps[11].pages[2].freelist)
    @test all(!, heap.size_heaps[11].pages[3].freelist)
    @test all(!, heap.size_heaps[11].pages[4].freelist)
    @test !heap.size_heaps[11].pages[5].freelist[1]
    @test all(heap.size_heaps[11].pages[5].freelist[2:end])
    for ptr in ptrs′
        HeapAllocator.free(heap, ptr)
        @test_throws ErrorException("free: double free") HeapAllocator.free(heap, ptr)
    end

    # malloc, realloc, free with custom type
    heap = HeapAllocator.Heap()
    ptr = HeapAllocator.malloc(heap, Int, 1024)
    @test ptr isa HeapAllocator.SizedPtr{Int}
    @test ptr.size == 1024 * sizeof(Int)
    ptr′ = HeapAllocator.realloc(heap, ptr, 2048)
    @test ptr′ isa HeapAllocator.SizedPtr{Int}
    @test ptr′.size == 2048 * sizeof(Int)
    @test_throws ErrorException("free: double free") HeapAllocator.free(heap, ptr)
    HeapAllocator.free(heap, ptr′)
    @test_throws ErrorException("free: double free") HeapAllocator.free(heap, ptr′)

    # malloc, realloc, free with arrays
    heap = HeapAllocator.Heap()
    x = HeapAllocator.alloc_array(heap, Int, 1024)
    x .= 1:1024
    x′ = HeapAllocator.realloc(heap, x, 2048)
    @test_throws ErrorException("free: double free") HeapAllocator.free(x)
    @test_throws ErrorException("free: double free") HeapAllocator.free(heap, x)
    @test_throws ErrorException("free: double free") HeapAllocator.free(heap, x.ptr)
    @test x′[1:1024] == 1:1024

    # Array functions
    heap = HeapAllocator.Heap()
    x = HeapAllocator.alloc_array(heap, Int, 8)
    x .= 1:8
    @test length(x) == 8
    x = HeapAllocator.resize(x, 0)
    @test length(x) == 0
    x = HeapAllocator.resize(x, 16)
    @test length(x) == 16
    @test x[1:8] == 1:8
    x = HeapAllocator.resize(x, 8)
    x = HeapAllocator.insert(x, 1, -1)
    x = HeapAllocator.insert(x, length(x) + 1, -1)
    x = HeapAllocator.insert(x, 2, -2)
    x = HeapAllocator.insert(x, length(x), -2)
    @test x == [-1; -2; 1:8; -2; -1]

    # n-d arrays
    heap = HeapAllocator.Heap()
    A = HeapAllocator.alloc_array(heap, Int, 64, 64)
    B = HeapAllocator.alloc_array(heap, Int, (64, 32))

    # Error paths
    heap = HeapAllocator.Heap()
    @test_throws ErrorException("pointer not malloc'd in this heap") HeapAllocator.free(heap, HeapAllocator.SizedPtr{UInt8}(0xbadc0ffee0ddf00d, 8 % UInt))
    HeapAllocator.malloc(heap, 8)
    @test_throws ErrorException("pointer not malloc'd in this heap") HeapAllocator.free(heap, HeapAllocator.SizedPtr{UInt8}(0xbadc0ffee0ddf00d, 8 % UInt))

    # Smoke show
    show(devnull, MIME"text/plain"(), heap)
    HeapAllocator.heap_stats(heap)

end
