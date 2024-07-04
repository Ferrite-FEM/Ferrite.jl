macro run_gpu(fun_expr, assembler, cellvalues, dh, colors)
    # Escaping arguments to handle their scope correctly
    local esc_fun_expr = esc(fun_expr)
    local esc_assembler = esc(assembler)
    local esc_cellvalues = esc(cellvalues)
    local esc_dh = esc(dh)
    local esc_colors = esc(colors)

    :(begin 
        local n_colors = length($esc_colors)
        for i in 1:n_colors
            local assembler_gpu = Adapt.adapt_structure(CUDA.KernelAdaptor(), $esc_assembler)
            local cellvalues_gpu = Adapt.adapt_structure(CUDA.KernelAdaptor(), $esc_cellvalues)
            local dh_gpu = Adapt.adapt_structure(CUDA.KernelAdaptor(), $esc_dh)
            local dh_gpu = Adapt.adapt_structure(CUDA.KernelAdaptor(), $esc_dh)
            local dh_gpu = Adapt.adapt_structure(CUDA.KernelAdaptor(), $esc_dh)

            # Configure the kernel
            local kernel = @cuda launch=false $esc_fun_expr(assembler_gpu, cellvalues_gpu, dh_gpu,length($esc_colors[i]), cu($esc_colors[i]))
            local config = launch_configuration(kernel.fun)
            local threads = min(length($esc_colors[i]), config.threads)
            local blocks = cld(length($esc_colors[i]), threads)

            # Launch the kernel
            kernel(assembler_gpu, cellvalues_gpu, dh_gpu,length($esc_colors[i]), cu($esc_colors[i]); threads, blocks)
        end
    end)
end
