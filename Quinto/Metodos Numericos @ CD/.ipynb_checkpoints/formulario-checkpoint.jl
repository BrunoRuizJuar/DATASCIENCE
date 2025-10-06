module formulario
using LinearAlgebra

### algoritmo 6.1
"""
    Resuelve el sistema de ecuaciones lineales Ax = b usando el método de eliminación gaussiana con sustitución hacia atrás.

    - A: matriz de coeficientes (n × n)
    - b: vector de términos independientes (n × 1)
    - Retorna: el vector solución x (n × 1)
"""
function gauss_elimination(A, b)

    # A es la matriz de coeficientes
    # b es el vector de términos independientes
    n = size(A, 1)

    # Construimos la matriz aumentada [A|b]
    Ab = hcat(A, b)

    # Eliminación hacia adelante
    for k in 1:n-1
        # Pivote: asegurarnos de que Ab[k,k] ≠ 0
        if Ab[k, k] == 0
            # Buscar fila con pivote no nulo más abajo
            for i in k+1:n
                if Ab[i, k] != 0
                    Ab[[k, i], :] = Ab[[i, k], :]   # intercambio de filas
                    break
                end
            end
        end

        # Eliminar debajo del pivote
        for i in k+1:n
            m = Ab[i, k] / Ab[k, k]   # multiplicador
            Ab[i, :] .= Ab[i, :] .- m .* Ab[k, :]
        end
    end

    # Sustitución hacia atrás
    x = zeros(Float64, n)
    for i in n:-1:1
        x[i] = (Ab[i, end] - sum(Ab[i, i+1:n] .* x[i+1:n])) / Ab[i, i]
    end

    return x
end

### algoritmo 6.2
"""
    Resuelve el sistema de ecuaciones lineales Ax = b usando el método de eliminación gaussiana con pivoteo parcial y sustitución hacia atrás.

    - A: matriz de coeficientes (n × n)
    - b: vector de términos independientes (n × 1)
    - Retorna: el vector solución x (n × 1)
"""
function gauss_elimination_pivot(A, b)
    n = size(A, 1)
    Ab = hcat(float.(A), float.(b))  # convertir a Float64
    # matriz aumentada [A|b]

    # Inicializamos NROW (vector de índices de filas)
    NROW = collect(1:n)

    # Proceso de eliminación
    for i in 1:n-1
        # Paso 3: encontrar fila p con pivote máximo en columna i
        p = i
        maxval = abs(Ab[NROW[i], i])
        for j in i+1:n
            if abs(Ab[NROW[j], i]) > maxval
                maxval = abs(Ab[NROW[j], i])
                p = j
            end
        end

        # Paso 4: verificar si el pivote es cero
        if maxval == 0
            error("No existe solución única (columna $i tiene pivote cero).")
        end

        # Paso 5: intercambio de índices de fila
        if NROW[i] != NROW[p]
            NCOPY = NROW[i]
            NROW[i] = NROW[p]
            NROW[p] = NCOPY
        end

        # Paso 6: eliminación hacia adelante
        for j in i+1:n
            m = Ab[NROW[j], i] / Ab[NROW[i], i]   # Paso 7
            Ab[NROW[j], i:end] .-= m .* Ab[NROW[i], i:end]  # Paso 8
        end
    end

    # Paso 9: verificar si último pivote es cero
    if Ab[NROW[n], n] == 0
        error("No existe solución única (último pivote cero).")
    end

    # Sustitución hacia atrás
    x = zeros(Float64, n)
    x[n] = Ab[NROW[n], end] / Ab[NROW[n], n]  # Paso 10

    for i in n-1:-1:1
        suma = sum(Ab[NROW[i], i+1:n] .* x[i+1:n])
        x[i] = (Ab[NROW[i], end] - suma) / Ab[NROW[i], i]  # Paso 11
    end

    return x  # Paso 12
end

### algoritmo 6.3
"""
    Resuelve el sistema de ecuaciones lineales Ax = b usando el método de eliminación gaussiana con pivoteo escalado y sustitución hacia atrás.

    - A: matriz de coeficientes (n × n)
    - b: vector de términos independientes (n × 1)
    - Retorna: el vector solución x (n × 1)
"""
function gauss_elimination_scaled(A, b)
    n = size(A, 1)
    Ab = hcat(float.(A), float.(b))        # [A|b] en Float64

    # s_i = máximo |a_ij| en cada fila (escala)
    s = [maximum(abs.(Ab[i, 1:n])) for i in 1:n]
    if any(si -> si == 0, s)
        error("No existe solución única (fila con todos ceros).")
    end

    # NROW: índices de filas
    NROW = collect(1:n)

    # Eliminación hacia adelante
    for i in 1:n-1
        # selección de pivote escalado
        p, maxratio = i, abs(Ab[NROW[i], i]) / s[NROW[i]]
        for j in i+1:n
            r = abs(Ab[NROW[j], i]) / s[NROW[j]]
            if r > maxratio
                p, maxratio = j, r
            end
        end
        # intercambio de índices
        if NROW[i] != NROW[p]
            NROW[i], NROW[p] = NROW[p], NROW[i]
        end
        # eliminación
        for j in i+1:n
            m = Ab[NROW[j], i] / Ab[NROW[i], i]
            Ab[NROW[j], i:end] .-= m .* Ab[NROW[i], i:end]
        end
    end

    if Ab[NROW[n], n] == 0
        error("No existe solución única (último pivote cero).")
    end

    # Sustitución hacia atrás
    x = zeros(Float64, n)
    x[n] = Ab[NROW[n], end] / Ab[NROW[n], n]
    for i in n-1:-1:1
        ssum = sum(Ab[NROW[i], i+1:n] .* x[i+1:n])
        x[i] = (Ab[NROW[i], end] - ssum) / Ab[NROW[i], i]
    end
    return x
end

###algoritmo 6.4
"""
    Resuelve el sistema de ecuaciones lineales Ax = b usando la factorización LU con sustitución hacia adelante y hacia atrás.

    - A: matriz de coeficientes (n × n)
    - b: vector de términos independientes (n × 1)
    - Retorna: el vector solución x (n × 1), y las matrices L y U
"""
# --- Factorización LU ---
function lu_factorization(A)
    n = size(A, 1)
    L = Matrix{Float64}(I, n, n)   # Identidad en Float64
    U = zeros(Float64, n, n)

    for i in 1:n
        # Calcular fila de U
        for j in i:n
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in 1:i-1; init=0.0)
        end

        # Calcular columna de L
        for j in i+1:n
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in 1:i-1; init=0.0)) / U[i, i]
        end
    end
    return L, U
end

# --- Sustitución hacia adelante ---
"""
    Resuelve el sistema de ecuaciones lineales Ly = b usando la sustitución hacia adelante.
"""
function forward_substitution(L, b)
    n = length(b)
    y = zeros(Float64, n)
    for i in 1:n
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in 1:i-1; init=0.0)) / L[i, i]
    end
    return y
end

# --- Sustitución hacia atrás ---
"""
    Resuelve el sistema de ecuaciones lineales Ux = y usando la sustitución hacia atrás.
"""
function backward_substitution(U, y)
    n = length(y)
    x = zeros(Float64, n)
    for i in n:-1:1
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in i+1:n; init=0.0)) / U[i, i]
    end
    return x
end

# --- Resolver sistema usando LU ---
"""
    Resuelve el sistema de ecuaciones lineales Ax = b usando la factorización LU.

    - A: matriz de coeficientes (n × n)
    - b: vector de términos independientes (n × 1)
    - Retorna: el vector solución x (n × 1), y las matrices L y U
"""
function solve_lu(A, b)
    L, U = lu_factorization(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x, L, U
end

end