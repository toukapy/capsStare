#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def list_datasets(h5_path: str):
    """Imprime los datasets disponibles dentro del .h5 (rutas completas y shapes)."""
    def _recurse(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"- {name}  shape={obj.shape} dtype={obj.dtype}")
    with h5py.File(h5_path, "r") as f:
        print(f"Datasets en {h5_path}:")
        f.visititems(_recurse)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convierte un array de imagen a uint8 [0,255] de forma segura."""
    a = np.asarray(arr)

    if a.dtype == np.uint8:
        return a

    if np.issubdtype(a.dtype, np.floating):
        vmin = float(np.nanmin(a))
        vmax = float(np.nanmax(a))
        # Caso típico [0,1]
        if vmin >= -1e-6 and vmax <= 1.0 + 1e-6:
            a = a * 255.0
        else:
            # Normaliza por min/max
            if vmax > vmin:
                a = (a - vmin) / (vmax - vmin) * 255.0
            else:
                a = np.zeros_like(a, dtype=np.float32)
        return np.clip(a, 0, 255).astype(np.uint8)

    # Otros enteros
    a = a.astype(np.float32)
    vmin = float(np.nanmin(a))
    vmax = float(np.nanmax(a))
    if vmax > vmin:
        a = (a - vmin) / (vmax - vmin) * 255.0
    else:
        a = np.zeros_like(a, dtype=np.float32)
    return np.clip(a, 0, 255).astype(np.uint8)


def save_images_from_h5(
    h5_path: str,
    out_dir: str,
    dataset_key: str,
    start: int = 0,
    limit: int | None = None,
    name_pattern: str = "{idx:06d}.png",
    assume_bgr: bool = True,
    skip_existing: bool = False,
):
    """
    Extrae imágenes de un dataset dentro de un .h5 y las guarda ordenadas.

    - h5_path: ruta al archivo .h5
    - out_dir: carpeta de salida (se crea si no existe)
    - dataset_key: ruta del dataset dentro del h5 (p.ej. 'face', 'images', 'data/frames')
    - start: índice inicial
    - limit: número máximo de imágenes a extraer (None = todas)
    - name_pattern: patrón de nombre, usa {idx}
    - assume_bgr: si True, convierte BGR→RGB (evita “tinte azul” típico de OpenCV)
    - skip_existing: si True, no sobreescribe si ya existe el archivo destino
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(
                f"Dataset '{dataset_key}' no encontrado. Usa --list para ver los disponibles."
            )
        dset = f[dataset_key]
        if dset.ndim < 2:
            raise ValueError(f"El dataset '{dataset_key}' no parece contener imágenes (shape={dset.shape})")

        N = dset.shape[0]  # asumimos primer eje = número de imágenes
        i0 = max(0, start)
        i1 = N if limit is None else min(N, i0 + max(0, limit))

        print(f"Extrayendo imágenes {i0}..{i1-1} de '{dataset_key}' (total {N})")
        print(f"Conversión BGR→RGB: {'ACTIVADA' if assume_bgr else 'desactivada'}")

        for i in range(i0, i1):
            out_name = name_pattern.format(idx=i)
            out_path = out / out_name
            if skip_existing and out_path.exists():
                continue

            arr = np.asarray(dset[i])  # (H,W,3) o (H,W) o (3,H,W), etc.

            # Reordenar ejes si vienen como (C,H,W) -> (H,W,C)
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))  # (H,W,C)

            # Escenarios: 2D (grises) o 3D con canales
            if arr.ndim == 2:
                img_u8 = to_uint8(arr)
                pil_img = Image.fromarray(img_u8, mode="L")
            elif arr.ndim == 3:
                if arr.shape[-1] == 1:
                    img_u8 = to_uint8(arr[..., 0])
                    pil_img = Image.fromarray(img_u8, mode="L")
                elif arr.shape[-1] == 3:
                    img_u8 = to_uint8(arr)
                    # Evitar “azul”: BGR -> RGB si procede
                    if assume_bgr:
                        img_u8 = img_u8[..., ::-1]
                    pil_img = Image.fromarray(img_u8, mode="RGB")
                else:
                    # Nº de canales inesperado: usa el primero como gris
                    img_u8 = to_uint8(arr[..., 0])
                    pil_img = Image.fromarray(img_u8, mode="L")
            else:
                raise ValueError(f"Forma de imagen no soportada: {arr.shape}")

            pil_img.save(out_path)

        print(f"Listo. Imágenes guardadas en: {out.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Extraer imágenes ordenadas desde un archivo .h5")
    p.add_argument("h5_path", type=str, help="Ruta al archivo .h5")
    p.add_argument("--dataset", "-d", type=str, default=None,
                   help="Clave del dataset dentro del .h5 (p.ej. 'face', 'images', 'data/frames')")
    p.add_argument("--out", "-o", type=str, required=False, help="Carpeta de salida")
    p.add_argument("--start", type=int, default=0, help="Índice inicial (por defecto 0)")
    p.add_argument("--limit", type=int, default=None, help="Número máximo de imágenes a extraer")
    p.add_argument("--pattern", type=str, default="{idx:06d}.png",
                   help="Patrón de nombre de archivo (usa {idx} para el índice)")
    p.add_argument("--list", action="store_true", help="Sólo listar datasets y salir")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--assume-bgr", action="store_true", help="Forzar BGR→RGB")
    g.add_argument("--assume-rgb", action="store_true", help="Forzar RGB (no convertir canales)")
    p.add_argument("--skip-existing", action="store_true", help="No sobrescribir archivos existentes")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_datasets(args.h5_path)
        return

    if args.dataset is None:
        print("❗ Debes indicar --dataset (usa --list para ver las opciones disponibles).")
        return

    if args.out is None:
        print("❗ Debes indicar --out para la carpeta de salida.")
        return

    # Por defecto asumimos BGR (común en ETH-XGaze / OpenCV)
    assume_bgr = True
    if args.assume_rgb:
        assume_bgr = False
    if args.assume_bgr:
        assume_bgr = True

    save_images_from_h5(
        h5_path=args.h5_path,
        out_dir=args.out,
        dataset_key=args.dataset,
        start=args.start,
        limit=args.limit,
        name_pattern=args.pattern,
        assume_bgr=assume_bgr,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
