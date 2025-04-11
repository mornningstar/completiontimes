import asyncio
import logging
import platform

from config.projects import PROJECTS
from src.data_handling.database.api_connection_async import APIConnectionAsync
from src.data_handling.clustering.file_cooccurence_analyser import FileCooccurrenceAnalyser
from src.data_handling.features.file_feature_engineering import FileFeatureEngineer
from src.predictions.file_model_trainer import FileModelTrainer
from src.visualisations.commit_visualiser import CommitVisualiser
from src.visualisations.file_visualiser import FileVisualiser

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('pymongo').setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Adjust the level as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

async def process_file_visualiser(api_connection, project_name, file_path, commit_visualiser, models, target,
                                        all_file_features, project, cluster_combined_df=None):
    if file_path:
        logging.info(f"Processing file: {file_path}, target: {target} in project: {project_name}")
    else:
        logging.info(f"Processing cluster, target: {target} in project: {project_name}")

    file_visualiser = FileVisualiser(
        api_connection,
        project_name,
        file_path,  # No specific file_path for cluster mode
        commit_visualiser,
        models,
        [target],
        all_file_features,
        cluster_combined_df=cluster_combined_df,
        )

    try:
        await file_visualiser.run()

        horizon = project['file_modeling_tasks'][target]['horizon']
        threshold = project['file_modeling_tasks'][target]['threshold']
        consecutive_days = project['file_modeling_tasks'][target]['consecutive_days']

        completion_date = await file_visualiser.predict_completion(target, horizon, threshold, consecutive_days)

        if completion_date:
            logging.info(f"Predicted completion date for {file_path}: {completion_date}")
        else:
            logging.info(f"No completion detected for {file_path} within {horizon} days.")

    except Exception as e:
        logging.error(f"Error processing file/cluster for target {target}. The error: {e}", exc_info=True)


async def process_project(project):
    project_name = project['name']
    models = project.get('models', [])
    file_modeling_tasks = project.get('file_modeling_tasks', {})
    modeling_tasks = project.get('modeling', [])
    recluster = project.get('recluster', True)
    replot = project.get('replot', False)
    plot_options = project.get('plot_options', {})

    api_connection = await APIConnectionAsync.create(project_name)

    task = "REGRESSION" # or "OLD"

    if task == "REGRESSION":
        try:
            logging.info(f"Starting processing for project: {project_name}")
            await api_connection.populate_db()

            feature_engineer = FileFeatureEngineer(api_connection, project_name, threshold=0.05, consecutive_days=14)
            file_features = await feature_engineer.run()

            commit_visualiser = CommitVisualiser(api_connection, project_name, models, modeling_tasks)
            await commit_visualiser.get_commits()

            file_model_trainer = FileModelTrainer(project_name, models[0])
            file_model_trainer.train_and_evaluate(file_features)

        except Exception:
            logging.exception('Error while processing project {}'.format(project_name))
        finally:
            logging.info('Project {} finished!'.format(project_name))
            await api_connection.close_session()
    else:

        try:
            logging.info(f"Starting processing for project: {project_name}")
            await api_connection.populate_db()

            feature_engineer = FileFeatureEngineer(api_connection, project_name, threshold=0.1, consecutive_days=14)

            all_file_features = await feature_engineer.run()

            commit_visualiser = CommitVisualiser(api_connection, project_name, models, modeling_tasks)
            await commit_visualiser.get_commits()

            cooccurrence_analyser = FileCooccurrenceAnalyser(
                commit_visualiser.commit_data, project_name, api_connection, all_file_features
            )

            cooccurrence_df, cooccurrence_categorized_df, proximity_df, cluster_combined_df = await (
                cooccurrence_analyser.run(recluster=recluster))

            if cluster_combined_df is None or cluster_combined_df.empty:
                logging.error(f"Cluster combined dataframe is None or empty for project {project_name}!")
            else:
                logging.info(f"Cluster combined dataframe preview:\n{cluster_combined_df.head()}")

            if replot:
                logging.info(f"Replotting enabled for project: {project_name}")
                cooccurrence_analyser.plot(
                    cooccurrence_df=cooccurrence_df,
                    cooccurrence_categorized_df=cooccurrence_categorized_df,
                    proximity_df=proximity_df,
                    combined_df=cluster_combined_df,
                    plot_options=plot_options
                )

            tasks = []
            for target, config in file_modeling_tasks.items():
                cluster_enabled = config.get('cluster', False)
                files = config.get('files', [])

                tasks.extend([
                    process_file_visualiser(
                        api_connection, project_name, file_path, commit_visualiser, models, target,
                        all_file_features, project
                    ) for file_path in files
                ])

                if cluster_enabled and cluster_combined_df is not None:
                    logging.info(f"I am inside the task. Clusters found: {cluster_combined_df['cluster'].unique()}")
                    tasks.extend([process_file_visualiser(
                        api_connection, project_name, None, commit_visualiser, models, target,
                        all_file_features, project, cluster_combined_df=cluster_combined_df
                    )]
                    )


            await asyncio.gather(*tasks)

        except Exception:
            logging.exception('Error while processing project {}'.format(project_name))

        finally:
            logging.info('Project {} finished!'.format(project_name))
            await api_connection.close_session()


async def main():
    tasks = [process_project(project) for project in PROJECTS]
    await asyncio.gather(*tasks)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())
