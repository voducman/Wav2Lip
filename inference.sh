python inference.py --checkpoint_path "/u01/manvd1/lipsync/Wav2Lip/checkpoints/wav2lip_gan.pth" --face "/u01/manvd1/lipsync/source_videos/sample/btv_480p.mp4" --audio "/u01/manvd1/lipsync/source_videos/sample/btv_noise_0.1.wav"  --wav2lip_batch_size 32 --face_det_batch_size 32 --outfile "/u01/manvd1/lipsync/output_videos/btv_480p_btv_noise.mp4"

python inference.py --checkpoint_path "/u01/manvd1/lipsync/Wav2Lip/checkpoints/wav2lip_gan.pth" --face "/u01/manvd1/lipsync/source_videos/sample/btv_hd.mp4" --audio "/u01/manvd1/lipsync/source_videos/sample/btv_noise_0.1.wav"  --wav2lip_batch_size 32 --face_det_batch_size 32 --outfile "/u01/manvd1/lipsync/output_videos/btv_hd_btv_noise.mp4"


python inference.py --checkpoint_path "/u01/manvd1/lipsync/Wav2Lip/checkpoints/wav2lip_gan.pth" --face "/u01/manvd1/lipsync/source_videos/sample/btv_fhd.mp4" --audio "/u01/manvd1/lipsync/source_videos/sample/btv_noise_0.1.wav"  --wav2lip_batch_size 32 --face_det_batch_size 32 --outfile "/u01/manvd1/lipsync/output_videos/btv_fhd_btv_noise.mp4"

