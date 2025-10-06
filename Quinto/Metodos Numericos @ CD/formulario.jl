module formulario
using LinearAlgebra, Printf, RowEchelon

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

# Función para imprimir matrices de forma legible
function mostrar_matriz(M, nombre="M")
    println("\n$nombre =")
    for fila in eachrow(M)
        println(join([@sprintf("%8.4f", x) for x in fila], " "))
    end
end

#6.6 Cholesky
"""
    cholesky_manual(A)

Implementa la descomposición de Cholesky de una matriz simétrica definida positiva A.
Devuelve la matriz triangular inferior L tal que A = L * L'.
"""
function cholesky_manual(A::Matrix{Float64})
    n = size(A, 1)
    L = zeros(Float64, n, n)

    for i in 1:n
        for j in 1:i
            suma = 0.0
            for k in 1:j-1
                suma += L[i, k] * L[j, k]
            end
            if i == j
                L[i, j] = sqrt(A[i, i] - suma)
            else
                L[i, j] = (A[i, j] - suma) / L[j, j]
            end
        end
    end
    return L
end

#6.7 Crout
"""
    resolver_tridiagonal_crout(l, d, u, b)

Resuelve un sistema de ecuaciones lineales tridiagonal Ax = b utilizando el
algoritmo de Crout optimizado.

# Argumentos
- `l::Vector{Float64}`: Vector de la subdiagonal (longitud n-1).
- `d::Vector{Float64}`: Vector de la diagonal principal (longitud n).
- `u::Vector{Float64}`: Vector de la superdiagonal (longitud n-1).
- `b::Vector{Float64}`: El vector del lado derecho del sistema (longitud n).

# Retorna
- `Vector{Float64}`: El vector de solución `x`.
"""
function resolver_tridiagonal_crout(l::Vector{Float64}, d::Vector{Float64}, u::Vector{Float64}, b::Vector{Float64})
    n = length(d)
    if !(length(l) == n - 1 && length(u) == n - 1 && length(b) == n)
        error("Las dimensiones de los vectores no son correctas para un sistema de n x n.")
    end

    # Vectores para almacenar los elementos calculados de L, U y z
    l_diag = zeros(n)
    u_super = zeros(n - 1)
    z = zeros(n)

    println("--- Iniciando Factorización de Crout y Sustitución Adelante ---")

    # --- Pasos 1-3: Configuran y resuelven Lz = b ---

    # Paso 1: Inicialización para i = 1
    l_diag[1] = d[1]
    u_super[1] = u[1] / l_diag[1]
    z[1] = b[1] / l_diag[1]

    # Paso 2: Bucle para i = 2, ..., n-1
    for i in 2:(n-1)
        # El subdiagonal de L es el mismo que el de A (lᵢ,ᵢ₋₁ = aᵢ,ᵢ₋₁)
        l_diag[i] = d[i] - l[i-1] * u_super[i-1]
        u_super[i] = u[i] / l_diag[i]
        z[i] = (b[i] - l[i-1] * z[i-1]) / l_diag[i]
    end

    # Paso 3: Cálculo final para i = n
    l_diag[n] = d[n] - l[n-1] * u_super[n-1]
    z[n] = (b[n] - l[n-1] * z[n-1]) / l_diag[n]

    println("Factorización y sustitución adelante completadas.")
    println("Vector intermedio z: ", round.(z, digits=4))

    # --- Pasos 4-5: Resuelven Ux = z ---

    println("\n--- Iniciando Sustitución Hacia Atrás ---")
    x = zeros(n)

    # Paso 4: Determinar xₙ
    x[n] = z[n]

    # Paso 5: Bucle hacia atrás para i = n-1, ..., 1
    for i in (n-1):-1:1
        x[i] = z[i] - u_super[i] * x[i+1]
    end

    # Paso 6: SALIDA
    println("Sustitución hacia atrás completada.")
    return x
end

#9.1 Metodo de la potencia
"""
    metodo_potencia(A, x, TOxL, N)

Calcula el eigenvalor dominante y su eigenvector asociado de una matriz A
usando el método de la potencia con norma-infinito.

# Argumentos
- `A::Matrix`: La matriz cuadrada de n x n.
- `x::Vector`: Un vector inicial no nulo de dimensión n.
- `TOL::Float64`: La tolerancia para el criterio de parada.
- `N::Int`: El número máximo de iteraciones permitidas.

# Devuelve
- `(μ, x)`: Una tupla con la aproximación del eigenvalor `μ` y el eigenvector `x`.
- Lanza un error si no converge o si el eigenvalor es cero.
"""
function metodo_potencia(A::Matrix, x::Vector, TOL::Float64, N::Int)
    # k es el contador de iteraciones
    k = 1

    # ---- Paso 2 y 3: Normalización inicial ----
    # Encontrar el índice del elemento con el máximo valor absoluto
    p = argmax(abs.(x))
    # Normalizar el vector x usando la norma-infinito (dividir por el máximo)
    x = x / x[p]

    # ---- Paso 4: Bucle principal ----
    while k <= N
        # ---- Paso 5: Multiplicar A por x ----
        y = A * x

        # ---- Paso 7 & 6: Encontrar el nuevo μ ----
        # La nueva aproximación del eigenvalor es el elemento de 'y'
        # con el máximo valor absoluto.
        μ = y[argmax(abs.(y))]

        # ---- Paso 8: Comprobar si el eigenvalor es cero ----
        if μ == 0
            error("La matriz tiene un eigenvalor de 0, el método no puede continuar.")
        end

        # ---- Paso 9 y 10: Calcular error y comprobar convergencia ----
        # El error es la diferencia (norma-infinito) entre el vector x anterior
        # y el nuevo vector normalizado (y / μ).
        err = norm(x - y / μ, Inf)

        # Actualizamos x para la siguiente iteración
        x = y / μ

        if err < TOL
            println("Convergencia alcanzada en la iteración $k.")
            return (μ, x) # Procedimiento exitoso
        end


        # ---- Paso 11: Incrementar el contador ----
        k += 1
    end

    # ---- Paso 12: Si se excede N ----
    error("El método no convergió en $N iteraciones.")
end

#9.2 Método potencia simetrica
"""
    metodo_potencia_simetrica(A, x, TOL, N)

Implementa el Método de la Potencia Simétrica para encontrar el eigenvalor 
dominante (de mayor magnitud) y su eigenvector asociado de una matriz simétrica.

# Argumentos
- `A::Matrix`: La matriz simétrica de n x n.
- `x::Vector`: Un vector inicial diferente de cero.
- `TOL::Float64`: La tolerancia para el criterio de convergencia.
- `N::Int`: El número máximo de iteraciones permitidas.

# Salida
- `(μ, x)`: Una tupla que contiene el eigenvalor aproximado `μ` y el 
  eigenvector normalizado `x`.
- Si el método no converge, imprime un mensaje de error y retorna la última 
  aproximación calculada.
- Si se encuentra un eigenvalor de 0, el proceso se detiene.
"""
function metodo_potencia_simetrica(A::Matrix{Float64}, x::Vector{Float64}, TOL::Float64, N::Int)

    # Verificación de que la matriz es simétrica
    if !issymmetric(A)
        error("La matriz de entrada A no es simétrica.")
    end

    println("--- Iniciando el Método de la Potencia Simétrica ---")

    # Paso 1: Inicialización
    k = 1
    # Normalizamos el vector inicial x
    x = x / norm(x)

    # Paso 2: Bucle principal de iteraciones
    while k <= N
        # Paso 3: Calcular y = Ax
        y = A * x

        # Paso 4: Aproximar el eigenvalor μ (Cociente de Rayleigh)
        μ = x' * y

        # Calculamos la norma de y para usarla en los siguientes pasos
        norm_y = norm(y)

        # Paso 5: Condición de parada si y es el vector nulo
        if norm_y == 0
            println("A tiene un eigenvalor de 0.")
            println("SALIDA: ('Eigenvector', x)")
            println("Seleccione un nuevo vector x y reinicie.")
            return # Se detiene la ejecución 
        end # Normalizamos el nuevo vector para el siguiente paso
        x_nuevo = y / norm_y # Paso 6: Calcular el error 
        # El error es la norma de la diferencia entre el vector actual y el nuevo 
        ERR = norm(x - x_nuevo) # Actualizamos x para la siguiente iteración 
        x = x_nuevo
        println("Iteración k: Eigenvalor ≈ ERR")

        # Paso 7: Criterio de convergencia
        if ERR < TOL
            println("\nProcedimiento fue exitoso.")
            println("Convergencia alcanzada en la iteración k.")
            return (μ, x) # SALIDA 
        end
        # Paso 8: Incrementar el contador de iteraciones 
        k += 1
    end
    # Paso 9: Mensaje si se excede el número de iteraciones 
    println("\nNúmero máximo de iteraciones (N) excedido.")
    println("El procedimiento no fue exitoso.")

    # Retornamos la última aproximación calculada
    return (μ, x)
end

# 9.5 Método de Householder
"""
    householder_tridiagonal(A)

Transforma una matriz simétrica A en una matriz tridiagonal simétrica T
que es similar a A (conserva los mismos eigenvalores) utilizando el
método de Householder.

# Argumentos
- `A::Matrix{Float64}`: La matriz simétrica n x n a transformar.

# Retorna
- `T::Matrix{Float64}`: La matriz tridiagonal simétrica resultante.
"""
function householder_tridiagonal(matrix::Matrix{Float64})
    # Hacemos una copia para no modificar la matriz original
    A = copy(matrix)
    n = size(A, 1)

    if size(A, 2) != n
        error("La matriz de entrada debe ser cuadrada.")
    end
    if !issymmetric(A)
        error("La matriz de entrada no es simétrica.")
    end

    println("--- Iniciando Método de Householder ---")

    # Paso 1: Bucle principal que itera sobre las columnas
    for k in 1:(n-2)
        # Paso 2: Calcular q
        q = dot(A[(k+1):n, k], A[(k+1):n, k]) # Suma de cuadrados

        # Paso 3: Calcular alpha
        if A[k+1, k] ≈ 0.0
            alpha = -sqrt(q)
        else
            alpha = -sqrt(q) * sign(A[k+1, k])
        end

        # Paso 4: Calcular RSQ
        RSQ = alpha^2 - alpha * A[k+1, k]

        # Si RSQ es cero, la columna ya está reducida. Continuamos.
        if RSQ ≈ 0.0
            continue
        end

        # Paso 5: Determinar el vector v
        v = zeros(n)
        v[k+1] = A[k+1, k] - alpha
        v[(k+2):n] = A[(k+2):n, k]

        # Paso 6: Determinar el vector u = (1/RSQ) * A*v
        u = (A * v) / RSQ

        # Paso 7: Calcular PROD = v' * u
        # Solo se necesita la parte de v que no es cero.
        PROD = dot(v, u)

        # Paso 8: Determinar el vector z = u - (PROD / 2RSQ) * v
        z = u - (PROD / (2 * RSQ)) * v

        # Paso 9: Actualizar la matriz A -> A - v*z' - z*v'
        # Esta operación matricial reemplaza los Pasos 10, 11 y 12.
        A -= v * z' + z * v'
    end

    # Limpieza final: debido a errores de punto flotante, los elementos que
    # deberían ser cero pueden ser números muy pequeños. Los forzamos a ser cero.
    for i in 1:n
        for j in 1:n
            if abs(i - j) > 1
                A[i, j] = 0.0
            end
        end
    end

    println("Tridiagonalización completada con éxito.")
    # Paso 15: SALIDA
    return A
end

#9.6 Método QR
"""
    metodo_qr(a_diag, b_subdiag; tol=1e-14, max_iter_total=10000)

Calcula los eigenvalores de una matriz simétrica tridiagonal usando
iteraciones QR con shift de Wilkinson y deflación.

- `a_diag` : diagonal principal (Vector{Float64})
- `b_subdiag` : subdiagonal (Vector{Float64}, longitud n-1)
"""
function metodo_qr(a_diag::Vector{Float64}, b_subdiag::Vector{Float64};
    tol::Float64=1e-14, max_iter_total::Int=10000)

    # Construimos la matriz densa a partir de la tridiagonal
    T = Matrix(Tridiagonal(b_subdiag, a_diag, b_subdiag))
    n = size(T, 1)
    eigenvalores = Float64[]
    iter_global = 0

    while n > 0
        # caso trivial
        if n == 1
            push!(eigenvalores, T[1, 1])
            break
        end

        converged_block = false

        for iter_local in 1:max_iter_total
            iter_global += 1

            # --- búsqueda de deflación (desde abajo) ---
            k = n - 1
            while k >= 1 && abs(T[k+1, k]) > tol * (abs(T[k, k]) + abs(T[k+1, k+1]))
                k -= 1
            end

            if k == n - 1
                # b[n-1] ≈ 0 → a[n] es eigenvalor
                push!(eigenvalores, T[n, n])
                n -= 1
                T = T[1:n, 1:n]
                converged_block = true
                break
            elseif k >= 1
                # Matrix se separa en 1..k y k+1..n -> resolvemos el bloque inferior recursivamente
                T_bottom = T[(k+1):n, (k+1):n]
                a_b = diag(T_bottom)
                b_b = [T_bottom[i+1, i] for i in 1:(size(T_bottom, 1)-1)]
                ev_b = metodo_qr_corregido(a_b, b_b; tol=tol, max_iter_total=max_iter_total)
                append!(eigenvalores, ev_b)

                # Reducimos al bloque superior y seguimos
                n = k
                T = T[1:n, 1:n]
                converged_block = true
                break
            end

            # --- no hay deflación local: hacemos una iteración QR con shift de Wilkinson ---
            a_nn1 = T[n-1, n-1]
            a_nn = T[n, n]
            b_nn1 = T[n, n-1]
            d = (a_nn1 - a_nn) / 2.0
            mu = a_nn - sign(d) * b_nn1^2 / (abs(d) + hypot(d, b_nn1))

            # Factorización QR de (T - mu*I)
            Ashift = T .- mu * Matrix{Float64}(I, n, n)
            F = qr(Ashift)            # factorización QR
            Q = Matrix(F.Q)
            R = Matrix(F.R)

            # Construimos la siguiente iteración: T_new = R * Q + mu * I
            T = R * Q .+ mu * Matrix{Float64}(I, n, n)

            # Forzamos simetría numérica para evitar pequeñas asimetrías por redondeo
            T = (T + T') / 2
        end

        if !converged_block
            error("No convergió el subproblema de tamaño $n después de $max_iter_total iteraciones.")
        end
    end

    return sort(eigenvalores)
end

#jacobi para valores y vectores propios
"""
    jacobi(A, b, x0, TOL, N)    
    Resuelve el sistema de ecuaciones lineales Ax = b usando el método iterativo de Jacobi.
    - A: matriz de coeficientes (n × n)
    - b: vector de términos independientes (n × 1)
    - x0: vector inicial (n × 1)
    - TOL: tolerancia para el criterio de paro
    - N: número máximo de iteraciones
    - Retorna: el vector solución x (n × 1)
"""
function jacobi(A, b, x0, TOL, N)
    n = length(b)           # tamaño del sistema
    x = copy(x0)            # x = x0 no crea un nuevo vector, por eso usamos copy
    x_new = similar(x0)     # nuevo vector del mismo tipo y tamaño que x0, pero sin copiar los valores

    # cálculo de cada componente
    for k in 1:N
        for i in 1:n
            suma = 0.0
            for j in 1:n
                if j != i
                    suma += A[i, j] * x[j]
                end
            end
            x_new[i] = (b[i] - suma) / A[i, i]
        end

        # criterio de paro
        if norm(x_new - x) < TOL
            println("El procedimiento fue exitoso en la iteración $k")
            return x_new
        end

        x .= x_new
    end

    println("Número máximo de iteraciones excedido. (El procedimiento no fue exitoso)")
    return x
end

#7.1 Tecnica iterativa de Jacobi
"""
    jacobi(A, b, x0, TOL, N)    
    Resuelve el sistema de ecuaciones lineales Ax = b usando el método iterativo de Jacobi.
    - A: matriz de coeficientes (n × n)
    - b: vector de términos independientes (n × 1)
    - x0: vector inicial (n × 1)
    - TOL: tolerancia para el criterio de paro
    - N: número máximo de iteraciones
    - Retorna: el vector solución x (n × 1)
"""
function jacobi_iterativo(A, b, x0, TOL, N)
    n = length(b)           # tamaño del sistema
    x = copy(x0)            # x = x0 no crea un nuevo vector, por eso usamos copy
    x_new = similar(x0)     # nuevo vector del mismo tipo y tamaño que x0, pero sin copiar los valores

    # cálculo de cada componente
    for k in 1:N
        for i in 1:n
            suma = 0.0
            for j in 1:n
                if j != i
                    suma += A[i, j] * x[j]
                end
            end
            x_new[i] = (b[i] - suma) / A[i, i]
        end

        # criterio de paro
        if norm(x_new - x) < TOL
            println("El procedimiento fue exitoso en la iteración $k")
            return x_new
        end

        x .= x_new
    end

    println("Número máximo de iteraciones excedido. (El procedimiento no fue exitoso)")
    return x
end

#7.2 Tecnica iterativa de Gauss-Seidel
"""
    metodo_gauss_seidel(A, b, x0; tol=1e-5, max_iter=100)

Resuelve el sistema de ecuaciones lineales Ax = b utilizando el método
iterativo de Gauss-Seidel, a partir de una aproximación inicial x0.

# Argumentos
- `A::Matrix{Float64}`: Matriz de coeficientes n x n.
- `b::Vector{Float64}`: Vector de términos independientes.
- `x0::Vector{Float64}`: Vector de aproximación inicial.

# Argumentos Opcionales
- `tol::Float64`: Tolerancia para el criterio de convergencia.
- `max_iter::Int`: Número máximo de iteraciones permitidas.

# Retorna
- `Vector{Float64}`: El vector de solución `x`, o la última aproximación si no converge.
"""
function metodo_gauss_seidel(A::Matrix{Float64}, b::Vector{Float64}, x0::Vector{Float64};
    tol::Float64=1e-5, max_iter::Int=100)

    n = size(A, 1)

    if any(isapprox.(diag(A), 0))
        error("Se encontró un cero en la diagonal. El método no es aplicable.")
    end

    x = copy(x0) # Vector que se actualizará en cada iteración

    println("--- Iniciando Método de Gauss-Seidel ---")

    # Paso 1: Inicializar el contador de iteraciones
    k = 1

    # Paso 2: Bucle principal de iteraciones
    while k <= max_iter
        # Guardamos una copia del vector ANTES de la iteración para el criterio de parada
        xo = copy(x)

        # Paso 3: Calcular cada componente de la nueva aproximación
        for i in 1:n
            # Suma de aᵢⱼ * xⱼ para j < i (usa los valores ya actualizados de x)
            suma1 = dot(A[i, 1:(i-1)], x[1:(i-1)])

            # Suma de aᵢⱼ * xⱼ para j > i (usa los valores de la iteración anterior)
            suma2 = dot(A[i, (i+1):n], xo[(i+1):n])

            # Despejamos y actualizamos x[i] inmediatamente
            x[i] = (b[i] - suma1 - suma2) / A[i, i]
        end

        # Paso 4: Criterio de convergencia
        error_estimado = norm(x - xo)
        println("Iteración ", k, ": ", error_estimado)

        if error_estimado < tol
            println("\nProcedimiento exitoso: Convergencia alcanzada en $k iteraciones.")
            return x # SALIDA
        end

        # Paso 5: Incrementar el contador
        k += 1

        # Paso 6: Actualizar XO ya se hizo implícitamente al inicio del bucle
    end

    # Paso 7: Mensaje si se excede el número de iteraciones
    println("\nProcedimiento no exitoso: Se excedió el número máximo de iteraciones ($max_iter).")
    return x
end

# Pseudo Inversa Moore-Penrose
"""
Calcula la pseudoinversa de Moore-Penrose para una matriz A utilizando el
algoritmo de factorización de rango completo.
"""
function pseudoinversa_full_rank(A::AbstractMatrix)
    # Convertimos a Float64 para compatibilidad con rref
    A_float = convert(Matrix{Float64}, A)

    # a) Reducir A a su forma escalonada reducida por filas (E_A)
    E_A = rref(A_float)

    # b) Formar la matriz B con las columnas pivote de A
    pivot_cols = []
    tol = 1e-8
    for i in 1:size(E_A, 1)
        pivot_index = findfirst(x -> abs(x) > tol, E_A[i, :])
        if pivot_index !== nothing
            push!(pivot_cols, pivot_index)
        end
    end
    unique!(sort!(pivot_cols))
    B = A_float[:, pivot_cols]

    # c) Formar la matriz C con las filas no nulas de E_A
    nonzero_rows = [i for i in 1:size(E_A, 1) if any(x -> abs(x) > tol, E_A[i, :])]
    C = E_A[nonzero_rows, :]

    # d) y e) Calcular A† con la fórmula final
    B_star = B'
    C_star = C'

    # Evitar error si una matriz es invertible y la otra no
    try
        inv_B_star_B = inv(B_star * B)
        inv_C_C_star = inv(C * C_star)
        return C_star * inv_C_C_star * inv_B_star_B * B_star
    catch e
        # Si falla, es probable que A sea invertible, usamos pinv como respaldo
        return pinv(A_float)
    end
end

#descomposición de valores singulares
function svd_eig(A_in; tol=1e-12, full=false)
    A = Array{Float64}(A_in)
    m, n = size(A)

    AtA = A' * A
    ev = eigen(AtA)
    vals = ev.values
    vecs = ev.vectors

    idx = sortperm(vals, rev=true)
    vals_sorted = vals[idx]
    Vfull = vecs[:, idx]

    s_all = sqrt.(clamp.(vals_sorted, 0.0, Inf))
    r = count(x -> x > tol, s_all)

    if r == 0
        U_comp = zeros(Float64, m, 0)
        s_comp = Float64[]
        V_comp = zeros(Float64, n, 0)
    else
        V_comp = Vfull[:, 1:r]
        s_comp = s_all[1:r]
        U_comp = (A * V_comp) ./ reshape(s_comp, 1, r)
    end

    if !full
        return U_comp, s_comp, V_comp
    end

    # construcción SVD completa
    # preparar Ufull
    if r == 0
        Ufull = qr!(copy(randn(m, m))).Q
    else
        Ufull = zeros(Float64, m, m)
        Ufull[:, 1:r] = U_comp
        function complete_orthonormal!(U::Matrix{Float64}, start_col::Int)
            m = size(U, 1)
            current = start_col - 1
            Qrand = qr!(copy(randn(m, m))).Q
            for j in 1:size(Qrand, 2)
                if current >= m
                    break
                end
                q = Qrand[:, j]
                if current > 0
                    proj = U[:, 1:current] * (U[:, 1:current]' * q)
                    q -= proj
                end
                nq = norm(q)
                if nq > 1e-12
                    current += 1
                    U[:, current] = q / nq
                end
            end
            if current < m
                Q2 = qr!(copy(randn(m, m))).Q
                col = current + 1
                for j in 1:size(Q2, 2)
                    if col > m
                        break
                    end
                    U[:, col] = Q2[:, j]
                    col += 1
                end
            end
            return nothing
        end

        if r < m
            complete_orthonormal!(Ufull, r + 1)
        end
    end

    Σ = zeros(Float64, m, n)
    for i in 1:min(m, n)
        Σ[i, i] = s_all[i]
    end

    return Ufull, Σ, Vfull
end

end # module formulario