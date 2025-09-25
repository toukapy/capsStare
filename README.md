# CapStARE: Casule-based Spatiotemporal Architecture for Robust and Efficient Gaze Estimation

https://github.com/user-attachments/assets/b40557ce-3e4e-430a-bae2-310dd96884e5

**CapStARE** is a capsule-based spatiotemporal model for gaze estimation. By combining capsule networks, attention mechanisms, and lightweight temporal decoders, CapStARE achieves state-of-the-art performance across multiple benchmarks while maintaining real-time efficiency. 

## 🔥 Key Features
- **Capsule-based spatial encoding** for robust handling of gaze under extreme head poses.
- **Dual-path GRU decoders** for temoral modeling of head and eye dynamics.
- **Lightweight design:** real-time inference at ~8ms per frame.
- **Generalizable**: validated across ETH-XGaze, MPIIfaceGaze, Gaze360, and RT-GENE.
- **Practical**: tested in real-time webcam video scenarios for human-robot interaction and unconstrained use cases.

## Benchmark Performance

| Method       | ETH-XGaze ↓ | MPIIFaceGaze ↓ | Gaze360 ↓ | RT-GENE ↓ | Params |
| ------------ | ----------- | -------------- | --------- | --------- | ------ |
| FullFace     | 7.38°       | 4.93°          | 14.99°    | 10.0°     | 196M   |
| Gaze360      | 4.46°       | 4.06°          | 11.04°    | 7.08°     | 11.9M  |
| GazeCapsNet  | 5.75°       | 4.06°          | **5.10°** | –         | 11.7M  |
| **CapStARE** | **3.36°**   | **2.65°**      | 9.06°     | **4.76°** | 13.0M  |

CapStARE achieves state-of-the-art accuracy on ETH-XGaze, MPIIFaceGaze, and RT-GENE, while remaining lightweight and real-time.

## Repository Structure

## Citation

## Contact

For questions, open an issue or reach out at miren.samaniego@ehu.eus


