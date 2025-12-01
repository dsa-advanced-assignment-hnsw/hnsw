import psycopg
import json
import time

file_path = '.cache/arxiv-metadata-oai-snapshot.json'

db_config = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "0909231769"
}

print("🚀 Bắt đầu tiến trình...")
start_time = time.time()

try:
    with psycopg.connect(**db_config) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS arxiv_papers")
            cur.execute("""
                CREATE TABLE arxiv_papers (
                    idx BIGSERIAL PRIMARY KEY,  -- Số thứ tự tự động (1, 2, 3...)
                    id TEXT UNIQUE,             -- ID gốc (bắt buộc duy nhất)
                    title TEXT,
                    abstract TEXT,
                    categories TEXT,
                    update_date TEXT
                )
            """)
            seen_ids = set()
            skipped = 0
            sql_copy = "COPY arxiv_papers (id, title, abstract, categories, update_date) FROM STDIN"
            with cur.copy(sql_copy) as copy:
                count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        current_id = item.get('id')
                        if current_id in seen_ids:
                            skipped += 1
                            continue
                        seen_ids.add(current_id)
                        copy.write_row((
                            current_id,
                            item.get('title', '').strip(),
                            item.get('abstract', '').strip(),
                            item.get('categories'),
                            item.get('update_date')
                        ))
                        count += 1
                        if count % 100000 == 0:
                            print(f"   -> Đã load {count} dòng...")
        conn.commit()
    print(f"🎉 Hoàn tất! Tổng cộng {count} dòng.")
    print(f"⚠️ Đã loại bỏ {skipped} dòng trùng ID.")
    print(f"⏱️ Thời gian: {time.time() - start_time:.2f} giây.")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")