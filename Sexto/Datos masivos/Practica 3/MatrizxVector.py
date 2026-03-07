from pyspark.sql import SparkSession

def main():
    # 1. Inicializamos la sesión de Spark
    spark = SparkSession.builder \
        .appName("MultiplicacionMatrizVectorMapReduce") \
        .getOrCreate()
    sc = spark.sparkContext

    # 2. Representación de los datos (Coordinate Format - COO)
    # Matriz M: (fila i, columna j, valor m_ij)
    datos_M = [
        (0, 0, 1.0), (0, 1, 2.0),
        (1, 0, 3.0), (1, 1, 4.0),
        (2, 0, 5.0), (2, 1, 6.0)
    ]
    
    # Vector v: (índice j, valor v_j)
    datos_v = [
        (0, 10.0),
        (1, 20.0)
    ]

    # Creamos los RDDs (Resilient Distributed Datasets)
    rdd_M = sc.parallelize(datos_M)
    rdd_v = sc.parallelize(datos_v)

    # ==========================================
    # FASE DE MAPEO (Alineación y Multiplicación)
    # ==========================================
    
    # Para simular los "k segmentos", mapeamos ambos RDDs usando 'j' como clave.
    # Spark internamente distribuirá (shuffling) los datos con la misma 'j' al mismo nodo.
    
    # RDD M transformado -> Clave: j, Valor: (i, m_ij)
    M_mapeado = rdd_M.map(lambda x: (x[1], (x[0], x[2])))
    
    # RDD v transformado -> Clave: j, Valor: v_j
    v_mapeado = rdd_v.map(lambda x: (x[0], x[1]))

    # Hacemos un JOIN por la clave 'j'
    # Resultado -> Clave: j, Valor: ((i, m_ij), v_j)
    datos_unidos = M_mapeado.join(v_mapeado)

    # Ahora aplicamos el mapeo de multiplicación: emitimos la fila 'i' como clave
    # y el producto parcial (m_ij * v_j) como valor.
    # Resultado -> Clave: i, Valor: m_ij * v_j
    productos_parciales = datos_unidos.map(lambda x: (x[1][0][0], x[1][0][1] * x[1][1]))

    # ==========================================
    # FASE DE REDUCCIÓN (Suma)
    # ==========================================
    
    # Agrupamos por la clave 'i' (la fila) y sumamos todos sus productos parciales
    vector_resultado = productos_parciales.reduceByKey(lambda a, b: a + b)

    # 3. Mostramos el resultado
    # Debería ser: Fila 0: (1*10 + 2*20) = 50, Fila 1: (3*10 + 4*20) = 110, Fila 2: (5*10 + 6*20) = 170
    print("Vector Resultante (i, valor):")
    for elemento in vector_resultado.sortByKey().collect():
        print(elemento)

    spark.stop()

if __name__ == "__main__":
    main()