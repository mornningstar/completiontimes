import platform

from src.data_handling.api_connection_async import APIConnectionAsync
from src.predictions.statistical_predictions.arima import ARIMAModel
from src.predictions.statistical_predictions.exponential_smoothing import SimpleExponentialSmoothing
from src.predictions.statistical_predictions.sarima import SARIMAModel
from src.visualisations.commit_visualiser import CommitVisualiser

import asyncio

from src.visualisations.file_visualiser import FileVisualiser


if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main():
    ohmyzsh_project = 'ohmyzsh/ohmyzsh'
    ohmyzsh_api = await APIConnectionAsync.create(ohmyzsh_project)

    try:
        #await ohmyzsh_api.populate_db()

        visualiser_ohmyzsh = CommitVisualiser(ohmyzsh_api.full_commit_info_collection)
        await visualiser_ohmyzsh.run(['totals', 'additions', 'deletions'])

        models = [
            ARIMAModel(),
            SimpleExponentialSmoothing()]

        file_paths = [
            'plugins/heroku/_heroku',
            'plugins/ubuntu/ubuntu.plugin.zsh',
            'plugins/mercurial/mercurial.plugin.zsh'
        ]

        for file_path in file_paths:
            visualiser_files = FileVisualiser(ohmyzsh_api.file_tracking_collection, file_path, models)
            await visualiser_files.run()

    finally:
        await ohmyzsh_api.close_session()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())
