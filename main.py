import platform

from api_connection_async import APIConnectionAsync
from commit_visualiser import CommitVisualiser

import asyncio

from file_visualiser import FileVisualiser

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main():
    vyvytn_KogRob22 = 'vyvytn/KogRob22'
    api_kogrob22 = await APIConnectionAsync.create(vyvytn_KogRob22)
    await api_kogrob22.populate_db()

    visualiser_KogRob22 = CommitVisualiser(api_kogrob22.full_commit_info_collection)
    await visualiser_KogRob22.fetch_data()
    visualiser_KogRob22.process_data()
    visualiser_KogRob22.plot_data(['totals', 'additions', 'deletions'])

    file_path = 'Abgabe 3/worlds/.humanoid_sprint.wbproj'
    visualiser_files = FileVisualiser(api_kogrob22.file_tracking_collection, file_path)
    await visualiser_files.fetch_data()
    visualiser_files.process_data()
    visualiser_files.plot_data()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())
