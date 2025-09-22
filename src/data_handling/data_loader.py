import pandas as pd

from src.data_handling.database.file_repo import FileRepository


class DataLoader:
    def __init__(self, file_repo: FileRepository):
        self.file_repo = file_repo

    async def fetch_all_files(self):
        all_files_data = await self.file_repo.get_all()

        rows = []
        for file_data in all_files_data:
            file_path = file_data['path']
            for commit in file_data.get('commit_history', []):
                rows.append({
                    "path": file_path,
                    "date": pd.to_datetime(commit['date']),
                    "size": commit['size'],
                    "committer": commit['committer'],
                    "lines_added": commit.get("additions", 0),
                    "lines_deleted": commit.get("deletions", 0),
                    "line_change": commit.get("total_changes", 0)
                })

        return pd.DataFrame(rows).sort_values('date')
