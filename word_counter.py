#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
英文单词学习助手
从文本文件中提取单词，录入SQLite数据库并统计出现次数
"""

import sqlite3
import re
import os
from collections import Counter
import argparse
import chardet
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import json


class WordCounter:
    def __init__(self, db_path="words.db"):
        """初始化单词计数器"""
        self.db_path = db_path
        self.progress_db_path = "progress.db"
        self.init_database()
        self.init_nltk()
    
    def init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建单词表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                count INTEGER DEFAULT 1,
                pos_tag TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建更新时间触发器
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_words_timestamp 
            AFTER UPDATE ON words
            BEGIN
                UPDATE words SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        ''')
        
        conn.commit()
        conn.close()
        
        # 初始化进度跟踪数据库
        self.init_progress_db()
        
        print(f"数据库已初始化: {self.db_path}")
    
    def init_progress_db(self):
        """初始化进度跟踪数据库"""
        conn = sqlite3.connect(self.progress_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_nltk(self):
        """初始化NLTK"""
        try:
            # 下载必要的NLTK数据
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            print("NLTK已初始化")
        except Exception as e:
            print(f"NLTK初始化失败: {e}")
            print("词性标注功能可能不可用")
    
    def extract_words(self, text):
        """从文本中提取英文单词"""
        # 使用正则表达式提取英文单词（不包括连字符）
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        # 转换为小写
        words = [word.lower() for word in words]
        return words
    
    def get_pos_tag(self, word):
        """获取单词的词性标注"""
        try:
            # 使用NLTK进行词性标注
            tagged = pos_tag([word])
            return tagged[0][1] if tagged else 'UNKNOWN'
        except Exception:
            return 'UNKNOWN'
    
    def detect_encoding(self, file_path):
        """检测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                # 如果检测置信度较低，尝试常见编码
                if confidence < 0.7:
                    for enc in ['utf-8', 'gbk', 'gb2312', 'big5', 'latin-1']:
                        try:
                            raw_data.decode(enc)
                            return enc
                        except UnicodeDecodeError:
                            continue
                
                return encoding if encoding else 'utf-8'
        except Exception:
            return 'utf-8'
    
    def process_text_file(self, file_path):
        """处理文本文件，提取单词并更新数据库"""
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return
        
        try:
            # 检测文件编码
            encoding = self.detect_encoding(file_path)
            
            # 尝试读取文件内容
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            
            # 提取单词
            words = self.extract_words(content)
            
            if not words:
                print(f"文件 {os.path.basename(file_path)} 未找到任何英文单词")
                return
            
            # 统计单词出现次数
            word_counts = Counter(words)
            
            # 更新数据库
            self.update_database(word_counts)
            
            print(f"处理完成！共找到 {len(word_counts)} 个不同的单词")
            print(f"总单词数: {sum(word_counts.values())}")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            print("跳过此文件，继续处理其他文件")
    
    def is_file_processed(self, file_path):
        """检查文件是否已被处理"""
        conn = sqlite3.connect(self.progress_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM processed_files WHERE file_path = ?", (file_path,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def mark_file_processed(self, file_path):
        """标记文件为已处理"""
        conn = sqlite3.connect(self.progress_db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO processed_files (file_path) VALUES (?)", (file_path,))
        conn.commit()
        conn.close()
    
    def process_directory(self, directory_path):
        """递归遍历目录，处理其中所有 .txt 文件"""
        if not os.path.exists(directory_path):
            print(f"目录不存在: {directory_path}")
            return
        
        # 收集所有txt文件
        all_txt_files = []
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                if file_name.lower().endswith('.txt'):
                    full_path = os.path.join(root, file_name)
                    all_txt_files.append(full_path)
        
        if not all_txt_files:
            print("未在目录中找到任何 .txt 文件")
            return
        
        # 过滤出未处理的文件
        unprocessed_files = [f for f in all_txt_files if not self.is_file_processed(f)]
        
        print(f"找到 {len(all_txt_files)} 个 .txt 文件")
        print(f"其中 {len(unprocessed_files)} 个文件需要处理")
        
        if not unprocessed_files:
            print("所有文件都已处理完成！")
            return
        
        # 处理未处理的文件
        processed_count = 0
        for i, file_path in enumerate(unprocessed_files, 1):
            print(f"\n==> 正在处理 ({i}/{len(unprocessed_files)}): {file_path}")
            self.process_text_file(file_path)
            self.mark_file_processed(file_path)
            processed_count += 1
        
        print(f"\n目录处理完成，本次处理了 {processed_count} 个新文件")
    
    def update_database(self, word_counts):
        """更新数据库中的单词计数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for word, count in word_counts.items():
            # 检查单词是否已存在
            cursor.execute("SELECT count, pos_tag FROM words WHERE word = ?", (word,))
            result = cursor.fetchone()
            
            if result:
                # 更新现有单词的计数
                new_count = result[0] + count
                cursor.execute(
                    "UPDATE words SET count = ? WHERE word = ?", 
                    (new_count, word)
                )
                print(f"更新单词 '{word}': {result[0]} + {count} = {new_count}")
            else:
                # 插入新单词，获取词性标注
                pos_tag = self.get_pos_tag(word)
                cursor.execute(
                    "INSERT INTO words (word, count, pos_tag) VALUES (?, ?, ?)", 
                    (word, count, pos_tag)
                )
                print(f"新增单词 '{word}': {count} (词性: {pos_tag})")
        
        conn.commit()
        conn.close()
    
    def show_stats(self, limit=20):
        """显示单词统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取总单词数
        cursor.execute("SELECT COUNT(*) FROM words")
        total_words = cursor.fetchone()[0]
        
        # 获取总出现次数
        cursor.execute("SELECT SUM(count) FROM words")
        total_count = cursor.fetchone()[0] or 0
        
        print(f"\n=== 单词统计 ===")
        print(f"不同单词数: {total_words}")
        print(f"总出现次数: {total_count}")
        
        # 显示最常见的单词
        cursor.execute("""
            SELECT word, count FROM words 
            ORDER BY count DESC, word ASC 
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        
        if results:
            print(f"\n=== 最常见的 {len(results)} 个单词 ===")
            print(f"{'单词':<20} {'次数':<10} {'百分比':<10}")
            print("-" * 40)
            
            for word, count in results:
                percentage = (count / total_count) * 100
                print(f"{word:<20} {count:<10} {percentage:.1f}%")
        
        conn.close()
    
    def search_word(self, word):
        """搜索特定单词"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT word, count, pos_tag, created_at, updated_at FROM words WHERE word = ?", (word.lower(),))
        result = cursor.fetchone()
        
        if result:
            word, count, pos_tag, created_at, updated_at = result
            print(f"\n单词: {word}")
            print(f"出现次数: {count}")
            print(f"词性: {pos_tag}")
            print(f"首次录入: {created_at}")
            print(f"最后更新: {updated_at}")
        else:
            print(f"未找到单词: {word}")
        
        conn.close()
    
    def export_words(self, output_file="words_export.txt", limit=None, pos_filter=None):
        """导出单词到文本文件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取所有单词的总出现次数
        cursor.execute("SELECT SUM(count) FROM words")
        all_words_total = cursor.fetchone()[0] or 0
        
        # 构建查询条件
        where_clause = ""
        params = []
        
        if pos_filter:
            where_clause = "WHERE pos_tag LIKE ?"
            params.append(f"%{pos_filter}%")
        
        # 获取筛选后的总出现次数用于计算词性内百分比
        if pos_filter:
            cursor.execute(f"SELECT SUM(count) FROM words {where_clause}", params)
            filtered_total = cursor.fetchone()[0] or 0
        else:
            filtered_total = all_words_total
        
        # 构建查询语句
        if limit:
            query = f"SELECT word, count, pos_tag FROM words {where_clause} ORDER BY count DESC, word ASC LIMIT ?"
            params.append(limit)
        else:
            query = f"SELECT word, count, pos_tag FROM words {where_clause} ORDER BY count DESC, word ASC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # 计算百分比总和
        filtered_percentage = 0
        all_percentage = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if pos_filter:
                f.write(f"单词\t次数\t词性\t词性百分比\t全部百分比 (筛选词性: {pos_filter})\n")
            else:
                f.write("单词\t次数\t词性\t词性百分比\t全部百分比\n")
            f.write("-" * 60 + "\n")
            for word, count, pos_tag in results:
                # 词性内百分比
                filtered_pct = (count / filtered_total) * 100 if filtered_total > 0 else 0
                # 全部单词百分比
                all_pct = (count / all_words_total) * 100 if all_words_total > 0 else 0
                
                filtered_percentage += filtered_pct
                all_percentage += all_pct
                f.write(f"{word}\t{count}\t{pos_tag}\t{filtered_pct:.2f}%\t{all_pct:.2f}%\n")
        
        print(f"单词已导出到: {output_file}")
        if pos_filter:
            print(f"筛选词性: {pos_filter}")
        if limit:
            print(f"导出了前 {len(results)} 个最常见的单词")
            print(f"词性内百分比总和: {filtered_percentage:.2f}%")
            print(f"全部单词百分比总和: {all_percentage:.2f}%")
        else:
            print(f"导出了所有 {len(results)} 个单词")
            print(f"词性内百分比总和: {filtered_percentage:.2f}%")
            print(f"全部单词百分比总和: {all_percentage:.2f}%")
        
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="英文单词学习助手")
    parser.add_argument("file", nargs="?", help="要处理的文本文件路径")
    parser.add_argument("--db", default="words.db", help="SQLite数据库文件路径")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--search", help="搜索特定单词")
    parser.add_argument("--export", help="导出单词到指定文件")
    parser.add_argument("--export-limit", type=int, help="导出前N个最常见的单词")
    parser.add_argument("--pos-filter", help="按词性筛选导出 (如: NN, VB, JJ等)")
    parser.add_argument("--limit", type=int, default=20, help="显示统计信息的单词数量限制")
    parser.add_argument("-all", "--all", action="store_true", help="遍历 ./book 目录下所有 .txt 文件")
    parser.add_argument("--book-dir", default="./book", help="当使用 --all 时要遍历的目录，默认 ./book")
    
    args = parser.parse_args()
    
    # 创建单词计数器
    counter = WordCounter(args.db)
    
    if args.all:
        counter.process_directory(args.book_dir)
    
    if args.file:
        # 处理文本文件
        counter.process_text_file(args.file)
    
    if args.stats:
        # 显示统计信息
        counter.show_stats(args.limit)
    
    if args.search:
        # 搜索单词
        counter.search_word(args.search)
    
    if args.export:
        # 导出单词
        counter.export_words(args.export, args.export_limit, args.pos_filter)
    
    # 如果没有指定任何操作，显示帮助信息
    if not any([args.file, args.stats, args.search, args.export]):
        print("英文单词学习助手")
        print("用法示例:")
        print("  python3 word_counter.py test.txt                    # 处理文本文件")
        print("  python3 word_counter.py --stats                    # 显示统计信息")
        print("  python3 word_counter.py --search hello             # 搜索单词")
        print("  python3 word_counter.py --export words.txt         # 导出所有单词")
        print("  python3 word_counter.py --export words.txt --export-limit 2000  # 导出前2000个单词")
        print("  python3 word_counter.py --export words.txt --pos-filter NN     # 导出名词(NN)单词")
        print("  python3 word_counter.py --export words.txt --pos-filter VB     # 导出动词(VB)单词")
        print("  python3 word_counter.py --all  --book-dir                      # 处理book目录下所有txt文件")


if __name__ == "__main__":
    main()
