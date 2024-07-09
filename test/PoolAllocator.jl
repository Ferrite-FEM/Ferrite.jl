using Test, Ferrite.PoolAllocator

@testset "PoolAllocator.jl" begin

    # Basic malloc, realloc, free
    mempool = PoolAllocator.MemoryPool{Int}()
    x = PoolAllocator.malloc(mempool, 1024)
    @test x isa PoolAllocator.PoolArray{Int}
    x .= 1:1024
    x′ = PoolAllocator.realloc(x, 2048)
    @test x′ isa PoolAllocator.PoolArray{Int}
    @test_throws ErrorException("free: block already free'd") PoolAllocator.free(x)
    @test x′[1:1024] == 1:1024
    PoolAllocator.free(x′)
    @test_throws ErrorException("free: block already free'd") PoolAllocator.free(x′)

    # Internal page allocation: exhaust some pages
    mempool = PoolAllocator.MemoryPool{Int}()
    xs = PoolAllocator.PoolArray{Int}[]
    for _ in 1:(PoolAllocator.PAGE_SIZE ÷ 512 ÷ sizeof(Int) * 2 + 1)
        x = PoolAllocator.malloc(mempool, 512)
        push!(xs, x)
    end
    @test length(mempool.books[10].pages) == 3
    @test all(!, mempool.books[10].pages[1].freelist)
    @test all(!, mempool.books[10].pages[2].freelist)
    @test !mempool.books[10].pages[3].freelist[1]
    @test all(mempool.books[10].pages[3].freelist[2:end])
    xs′ = PoolAllocator.PoolArray{Int}[]
    for x in xs
        x′ = PoolAllocator.realloc(x, 1024)
        @test_throws ErrorException("free: block already free'd") PoolAllocator.free(x)
        push!(xs′, x′)
    end
    @test length(mempool.books[10].pages) == 3 # TODO
    @test all(mempool.books[10].pages[1].freelist)
    @test all(mempool.books[10].pages[2].freelist)
    @test all(mempool.books[10].pages[3].freelist)
    @test length(mempool.books[11].pages) == 5
    @test all(!, mempool.books[11].pages[1].freelist)
    @test all(!, mempool.books[11].pages[2].freelist)
    @test all(!, mempool.books[11].pages[3].freelist)
    @test all(!, mempool.books[11].pages[4].freelist)
    @test !mempool.books[11].pages[5].freelist[1]
    @test all(mempool.books[11].pages[5].freelist[2:end])
    for x in xs′
        PoolAllocator.free(x)
        @test_throws ErrorException("free: block already free'd") PoolAllocator.free(x)
    end
    PoolAllocator.free(mempool)
    @test length(mempool.books) == 0

    # Array functions
    mempool = PoolAllocator.MemoryPool{Int}()
    x = PoolAllocator.malloc(mempool, 8)
    @test length(x) == 8
    x = PoolAllocator.resize(x, 0)
    @test length(x) == 0
    x = PoolAllocator.resize(x, 16)
    @test length(x) == 16
    x .= 1:16
    x = PoolAllocator.resize(x, 8)
    @test x == 1:8
    x = PoolAllocator.resize(x, 8)
    x = PoolAllocator.insert(x, 1, -1)
    x = PoolAllocator.insert(x, length(x) + 1, -1)
    x = PoolAllocator.insert(x, 2, -2)
    x = PoolAllocator.insert(x, length(x), -2)
    @test x == [-1; -2; 1:8; -2; -1]

    # n-d arrays
    mempool = PoolAllocator.MemoryPool{Int}()
    A = PoolAllocator.malloc(mempool, 64, 64)
    @test size(A) == (64, 64)
    B = PoolAllocator.malloc(mempool, (64, 32))
    @test size(B) == (64, 32)

    # Smoke test for `show`
    show(devnull, MIME"text/plain"(), mempool)

end
