import pymysql
from datetime import datetime

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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    conn.close()


def salvar_metrica(quantidade, tempo_estimado, entradas, saidas):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO historico_filas (quantidade, tempo_estimado, entradas, saidas) VALUES (%s, %s, %s, %s)",
            (quantidade, tempo_estimado, entradas, saidas),
        )
    conn.commit()
    conn.close()


def listar_historico(limite=500):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM historico_filas ORDER BY created_at DESC LIMIT %s", (limite,)
        )
        rows = cur.fetchall()
    conn.close()
    return rows


def estatisticas():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS total_registros,
                ROUND(AVG(quantidade), 1) AS media_pessoas,
                MAX(quantidade) AS max_pessoas,
                MIN(quantidade) AS min_pessoas,
                ROUND(AVG(tempo_estimado), 1) AS media_tempo,
                MAX(tempo_estimado) AS max_tempo,
                SUM(entradas) AS total_entradas,
                SUM(saidas) AS total_saidas
            FROM historico_filas
        """)
        stats = cur.fetchone()

        cur.execute(
            "SELECT quantidade, created_at FROM historico_filas ORDER BY created_at ASC"
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
