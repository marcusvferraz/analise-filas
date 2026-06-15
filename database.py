import pymysql

DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASS = ""
DB_NAME = "analise_filas"


def get_conn():
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME,
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
    )


def criar_tabela():
    conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, charset="utf8mb4")
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4")
    conn.close()

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS historico_filas (
                id INT AUTO_INCREMENT PRIMARY KEY,
                quantidade INT NOT NULL,
                tempo_estimado INT NOT NULL,
                entradas INT DEFAULT 0,
                saidas INT DEFAULT 0,
                tempo_real_medio DECIMAL(10,1) DEFAULT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    conn.close()


def salvar_metrica(quantidade, tempo_estimado, entradas, saidas, tempo_real_medio=None):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO historico_filas (quantidade, tempo_estimado, entradas, saidas, tempo_real_medio) "
            "VALUES (%s, %s, %s, %s, %s)",
            (quantidade, tempo_estimado, entradas, saidas, tempo_real_medio),
        )
    conn.commit()
    conn.close()


def listar_historico(pagina=1, por_pagina=25):
    conn = get_conn()
    offset = (pagina - 1) * por_pagina
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM historico_filas ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (por_pagina, offset),
        )
        rows = cur.fetchall()
    conn.close()
    return rows


def contar_historico():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS total FROM historico_filas")
        total = cur.fetchone()["total"]
    conn.close()
    return total


def estatisticas():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS total_registros,
                ROUND(AVG(quantidade), 1) AS media_pessoas,
                MAX(quantidade) AS max_pessoas,
                MIN(quantidade) AS min_pessoas,
                ROUND(AVG(tempo_estimado), 1) AS media_tempo_est,
                MAX(tempo_estimado) AS max_tempo_est,
                ROUND(AVG(tempo_real_medio), 1) AS media_tempo_real,
                MAX(tempo_real_medio) AS max_tempo_real,
                SUM(entradas) AS total_entradas,
                SUM(saidas) AS total_saidas
            FROM historico_filas
        """)
        stats = cur.fetchone()

        cur.execute(
            "SELECT quantidade, tempo_real_medio, created_at FROM historico_filas ORDER BY created_at ASC"
        )
        crescimento = cur.fetchall()
    conn.close()

    if crescimento:
        primeiro = crescimento[0]["quantidade"]
        ultimo = crescimento[-1]["quantidade"]
        stats["crescimento"] = ultimo - primeiro
        stats["dados_crescimento"] = crescimento
    else:
        stats["crescimento"] = 0
        stats["dados_crescimento"] = []

    return stats
