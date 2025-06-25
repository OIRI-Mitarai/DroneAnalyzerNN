import os

def delete_first_n_lines_from_csv(folder_path, n=100):
    """
    指定されたフォルダ内のすべてのCSVファイルの先頭n行を削除します。

    Args:
        folder_path (str): CSVファイルが存在するフォルダのパス。
        n (int): 削除する行数（デフォルトは100）。
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()

                if len(lines) > n:
                    with open(filepath, 'w', encoding='utf-8') as outfile:
                        outfile.writelines(lines[n:])
                    print(f"ファイル '{filename}' の先頭 {n} 行を削除しました。")
                else:
                    print(f"ファイル '{filename}' は {n} 行未満のため、処理を行いませんでした。")

            except Exception as e:
                print(f"ファイル '{filename}' の処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
    folder_path = "test"  # 処理対象のフォルダ名
    delete_first_n_lines_from_csv(folder_path)
    print("処理が完了しました。")
