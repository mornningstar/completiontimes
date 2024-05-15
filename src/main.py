import asyncio
import platform

from config.projects import PROJECTS
from src.data_handling.api_connection_async import APIConnectionAsync
from src.visualisations.commit_visualiser import CommitVisualiser
from src.visualisations.file_visualiser import FileVisualiser

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def process_project(project):
    project_name = project['name']
    models = project['models']
    file_paths = project['file_paths']
    commit_views = project['commit_views']

    api_connection = await APIConnectionAsync.create(project_name)

    try:
        #await api_connection.populate_db()
        visualiser = CommitVisualiser(api_connection.full_commit_info_collection)
        await visualiser.run(commit_views)

        for file_path in file_paths:
            visualiser_files = FileVisualiser(api_connection.file_tracking_collection, file_path, models)
            await visualiser_files.run()

    finally:
        await api_connection.close_session()


async def main():
    tasks = [process_project(project) for project in PROJECTS]
    await asyncio.gather(*tasks)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())
