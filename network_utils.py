from simple_youtube_api.Channel import Channel
from simple_youtube_api.LocalVideo import LocalVideo
from db import models


def upload_youtube_video(translation: models.Translation, desc="CrossLingual Demo"):
    src_media_path = translation.cross_project.get_media_path()
    mux_media_path = src_media_path.with_suffix('.output.mp4')
    channel = Channel()
    channel.login("client_secret.json", "credentials.storage")

    # setting up the video that is going to be uploaded
    video = LocalVideo(file_path=mux_media_path)

    # setting snippet
    video.set_title(translation.cross_project.title)
    video.set_description(desc)
    # video.set_tags(["this", "tag"])
    # video.set_category("gaming")
    video.set_default_language(translation.lang)

    # setting status
    video.set_embeddable(True)
    video.set_license("creativeCommon")
    video.set_privacy_status("unlisted")
    video.set_public_stats_viewable(True)

    # setting thumbnail
    # video.set_thumbnail_path('test_thumb.png')
    # uploading video and printing the results
    # liking video
    # video.like()
    try:
        video = channel.upload_video(video)
        return f"https://youtu.be/{video.id}"
    except:
        return ""


if __name__ == '__main__':
    from sqlalchemy.orm import Session
    from db import crud, database

    db: Session = database.SessionLocal()
    user = crud.get_user_by_email(db, "taras.y.sereda@proton.me")
    project_name = "espanol-con-juan"
    translation_db = crud.get_translation_by_title_and_lang(db, project_name, "EN-US", user.id)
    print(translation_db)
    youtube_link = upload_youtube_video(translation_db)
    print(youtube_link)
