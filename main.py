from drums_SAE.dataset_prepare import create_dataset_manifest, encode_audio

if __name__ == "__main__":
    create_dataset_manifest(root_path="data", output_path="data/GT/drums_manifest.csv")
    encode_audio(
        manifest_csv="data/GT/drums_manifest.csv",
        save_path="data/drums_encoded.npz",
    )
