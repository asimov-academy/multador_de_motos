import os
import re
from pytube import YouTube, Playlist, Channel

class YoutubeDownloader:
    def download_video(self, link='https://www.youtube.com/watch?v=dQw4w9WgXcQ'):
        print(f"Baixando vídeo do link: {link}")
        yt = YouTube(link)
        title = yt.title
        stream = yt.streams.get_highest_resolution()
        

        output_path = 'videos/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Criada pasta '{output_path}'")

        title = re.sub(r'[<>:"/\\|?*]', '', title)
        title = title.replace("'", "")
        title = title.strip().replace(' ', '_')
        
        video_path = stream.download(output_path, f'{title}.mp4')
        print(f"Baixado vídeo '{title}' para '{video_path}'")
        video_path = f'{output_path}/{title}'
        print(f"Vídeo baixado em: {video_path}")
        return video_path
    
    def download_playlist(self, playlist_url):
        """Create a Playlist object with the provided URL"""
        playlist = Playlist(playlist_url)
        playlist_title = playlist.title
        print(f'Downloading playlist: {playlist_title}')

        output_path = f'videos/{playlist_title}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Criada pasta '{output_path}'")
        
        for video in playlist.videos:
            title = video.title
            title = re.sub(r'[<>:"/\\|?*]', '', title)
            title = title.replace("'", "")
            title = title.strip().replace(' ', '_')
            
            print(f'Downloading video: {video.title}')
            video.streams.get_highest_resolution().download(output_path, title)
        
        print('All videos downloaded!')

    def download_channel(self, channel_url):
        """Create a Playlist object with the provided URL"""
        channel = Channel(channel_url)
        channel_title = channel.channel_name
        print(f'Downloading channel: {channel_title}')

        output_path = f'videos/channel/{channel_title}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Criada pasta '{output_path}'")
        
        for video in channel.videos:
            title = video.title
            title = re.sub(r'[<>:"/\\|?*]', '', title)
            title = title.replace("'", "")
            title = title.strip().replace(' ', '_')
            title = f'{title}.mp4'
            
            print(f'Downloading video: {video.title}')
            video.streams.get_highest_resolution().download(output_path, title)
        
        print(f'All videos from {channel_title} downloaded!')



if __name__ == '__main__':
    yt = YoutubeDownloader()

    command = input(
        '''
        press \n
        1 for direct links \n
        "P" for a playlist \n
        "C" for channels \n
        or type a search query \n
        '''
        )
    if command == "1":
        links = [
            # 'https://www.youtube.com/watch?v=V8Ld3cljsNE',
            # 'https://www.youtube.com/watch?v=-Y4SJmIWvAc&t=2282s',
            'https://www.youtube.com/watch?v=V8Ld3cljsNE'
            ]
        for link in links:
            yt.download_video(link)

    elif command == 'p':
        link = 'https://www.youtube.com/playlist?list=PLha-P1s8bIg_UJXz8ESxFhhaDHuXkiy_F'
        yt.download_playlist(link)

    elif command == 'c':
        link = 'https://www.youtube.com/@AsimovAcademy'
        yt.download_channel(link)
