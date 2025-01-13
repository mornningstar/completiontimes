import asyncio
import logging
import platform

from config.projects import PROJECTS
from src.data_handling.api_connection_async import APIConnectionAsync
from src.data_handling.file_cooccurence_analyser import FileCooccurenceAnalyser
from src.data_handling.file_feature_engineering import FileFeatureEngineer
from src.visualisations.commit_visualiser import CommitVisualiser
from src.visualisations.file_visualiser import FileVisualiser

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Adjust the level as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


async def process_project(project):
    project_name = project['name']
    models = project.get('models', [])
    file_modeling_tasks = project.get('file_modeling_tasks', {})
    modeling_tasks = project.get('modeling', [])

    api_connection = await APIConnectionAsync.create(project_name)

    try:
        logging.info(f"Starting processing for project: {project_name}")
        await api_connection.populate_db()

        # Centralised feature engineering
        logging.info(f"Running feature engineering for project: {project_name}")
        feature_engineer = FileFeatureEngineer(api_connection, project_name)
        all_file_features = await feature_engineer.run()  # Calculate features for all files

        commit_visualiser = CommitVisualiser(api_connection, project_name, models, modeling_tasks)
        await commit_visualiser.get_commits()

        #if modeling_tasks:
            #await commit_visualiser.run()
            #await repodata_handling.plot()

        cluster_combined_df = None
        if any(config.get('cluster', False) for config in file_modeling_tasks.values()):
            cooccurrence_analyser = FileCooccurenceAnalyser(
                commit_visualiser.commit_data, project_name, api_connection, all_file_features
            )
            cluster_combined_df = cooccurrence_analyser.run()

        for target, config in file_modeling_tasks.items():
            cluster_enabled = config.get('cluster', False)
            files = config.get('files', [])

            for file_path in files:
                visualiser_files = FileVisualiser(
                    api_connection,
                    project_name,
                    file_path,
                    commit_visualiser,
                    models,
                    [target],  # Target-specific tasks
                    all_file_features
                )

                logging.info(f"Processing file: {file_path}, target: {target} in project: {project_name}")
                await visualiser_files.run(mode="file")

            if cluster_enabled and cluster_combined_df is not None:
                visualiser_clusters = FileVisualiser(
                    api_connection,
                    project_name,
                    None,  # No specific file_path for cluster mode
                    commit_visualiser,
                    models,
                    [target],
                    all_file_features,
                    cluster_combined_df=cluster_combined_df,  # Pass the precomputed clusters
                )

                logging.info(f"Processing clusters for target: {target} in project: {project_name}")
                await visualiser_clusters.run(mode="cluster")


    except Exception:
        logging.exception('Error while processing project {}'.format(project_name))

    finally:
        logging.info('Project {} finished successfully!'.format(project_name))
        await api_connection.close_session()


async def main():
    tasks = [process_project(project) for project in PROJECTS]
    await asyncio.gather(*tasks)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())
