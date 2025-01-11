import asyncio
import platform
import logging


from config.projects import PROJECTS
from src.data_handling.api_connection_async import APIConnectionAsync
from src.data_handling.file_cooccurence_analyser import FileCooccurenceAnalyser
from src.data_handling.repodata_handler import RepoDataHandler
from src.visualisations.commit_visualiser import CommitVisualiser
from src.visualisations.file_visualiser import FileVisualiser

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Adjust the level as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


async def process_project(project):
    project_name = project['name']
    models = project['models'] if 'models' in project else []
    file_paths = project['file_paths'] if 'file_paths' in project else []
    file_modeling_tasks = project['file_modeling_tasks'] if 'file_modeling_tasks' in project else []
    modeling_tasks = project['modeling'] if 'modeling' in project else []

    api_connection = await APIConnectionAsync.create(project_name)

    try:
        await api_connection.populate_db()

        if modeling_tasks:
            commit_visualiser = CommitVisualiser(api_connection, project_name, models, modeling_tasks)
            await commit_visualiser.run()
            #await repodata_handling.plot()

            #cooccurrence_analyser = FileCooccurenceAnalyser(commit_visualiser.commits, project_name)
            #cooccurrence_analyser.run()

        if file_paths:
            for file_path in file_paths:
                visualiser_files = FileVisualiser(api_connection, project_name, file_path, models, file_modeling_tasks)
                await visualiser_files.run()

    finally:
        logging.info('Project {} finished successfully!'.format(project_name))
        await api_connection.close_session()


async def main():
    tasks = [process_project(project) for project in PROJECTS]
    await asyncio.gather(*tasks)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())
