from pytubefix import YouTube

def download_youtube_video(url, save_path='.'):
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Print video title
        print(f"Title: {yt.title}")

        # Select the highest resolution stream
        stream = yt.streams.get_highest_resolution()

        # Download the video
        print(f"Downloading {yt.title}...")
        stream.download(output_path=save_path, filename="video.mp4")
        print(f"Download completed! Video saved to {save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # URL of the YouTube video
    video_url = 'https://www.youtube.com/watch?v=sa-NRvskyMc'
    
    # Path where you want to save the video
    save_directory = '.'  # Current directory; change this if needed
    
    download_youtube_video(video_url, save_directory)