# This is a sample Python script.
from api_connection_async import APIConnectionAsync
from commit_visualiser import CommitVisualiser

import asyncio


async def main():
    vyvytn_KogRob22 = 'vyvytn/KogRob22'
    api_kogrob22 = await APIConnectionAsync.create(vyvytn_KogRob22)
    await api_kogrob22.populate_db()

    visualiser_KogRob22 = CommitVisualiser(api_kogrob22.full_commit_info_collection)
    await visualiser_KogRob22.fetch_data()
    visualiser_KogRob22.process_data()
    visualiser_KogRob22.plot_data(['totals', 'additions', 'deletions'])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
