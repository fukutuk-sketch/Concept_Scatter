# Concept Scatter — Claude Code 作業ルール

最初に `PROJECT_CONTEXT.md` を読むこと（仕様・状態・未確認事項はそこにある）。

- 作者は日本語でやりとりする非プログラマー。UI のラベルとツールチップは日本語で付ける。
- アドオン本体は `concept_scatter.py` の 1 ファイル。`concept_ground_paint.py` は参照用で、編集しない。
- 新しく作るデータ名は `CSC_` で始める。地面ペイント（`CGP_*`）など他ツールのデータは読むだけで書かない。
- 変更したら `python3 -m py_compile concept_scatter.py` と `python3 tests/mock_check.py` を通す。
  Blender（`bpy`）はクラウドでは使えないので、ノードや描画・実際の操作は実機での確認項目として
  `PROJECT_CONTEXT.md` の「すぐやること」に書き残す。「確認済み」と書くのは実機で見たものだけ。
- 使うアイコン名・プロパティ・演算子は存在を確かめる（無効なものはパネルを壊す）。
