"""Plot declared boundary patches on the actual h=3 mm source-cup surface mesh."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import meshio
import numpy as np

HERE = Path(__file__).resolve().parent
X, Z = -4.0, 275.0


def patches(points, triangles):
    p = points[triangles]
    c = p.mean(axis=1)
    r = np.linalg.norm(c[:, (0, 2)] - [X, Z], axis=1)
    area = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1) / 2
    bearing = (np.abs(r-35) < .35) & (c[:, 1] >= 21-.1) & (c[:, 1] <= 26+.1)
    shoulder = (np.abs(c[:, 1]-26) < .08) & (r >= 35-.2) & (r <= 36.3)
    # Six source front-inner screws lie on Ø27, transformed into world X/Z.
    source_xy = np.array([[-11.69134295109,-6.75], [0,-13.5], [11.69134295109,-6.75],
                          [11.69134295109,6.75], [0,13.5], [-11.69134295109,6.75]])
    centers = np.column_stack((X+source_xy[:, 1], Z-source_xy[:, 0]))
    hole_distance = np.linalg.norm(c[:, None, (0, 2)]-centers[None, :, :], axis=2).min(axis=1)
    motor = (c[:, 1] >= 54.8) & (c[:, 1] <= 59.2) & (np.abs(hole_distance-1.65) < .3)
    flange = (np.abs(c[:, 2]-140) < .12) & (c[:, 1] >= 45-.1) & (c[:, 1] <= 51+.1)
    return c, area, {'bearing_radial': bearing, 'bearing_axial_shoulder': shoulder,
                     'motor_bolt_tangential': motor, 'distal_flange_load': flange}


def main():
    mesh = meshio.read(HERE / 'cup-h3.msh')
    tri = np.vstack([block.data for block in mesh.cells if block.type == 'triangle'])
    c, area, groups = patches(mesh.points, tri)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    ax = axes[0]
    ax.scatter(c[:, 1], c[:, 2], s=.25, c='0.8', rasterized=True)
    colors = {'bearing_radial':'#0066a1','bearing_axial_shoulder':'#00a78e',
              'motor_bolt_tangential':'#ad38b9','distal_flange_load':'#ec711a'}
    for name, mask in groups.items():
        ax.scatter(c[mask, 1], c[mask, 2], s=6, label=name.replace('_',' '), c=colors[name], rasterized=True)
    ax.set(xlabel='world Y (mm)', ylabel='world Z (mm)', title='Y–Z projection')
    ax.set_aspect('equal')
    ax = axes[1]
    ax.scatter(c[:, 0], c[:, 2], s=.25, c='0.83', rasterized=True)
    for name, mask in groups.items():
        ax.scatter(c[mask, 0], c[mask, 2], s=6, c=colors[name], rasterized=True)
    ax.set(xlabel='world X (mm)', ylabel='world Z (mm)', title='X–Z projection; axis X=−4, Z=275 mm')
    ax.set_aspect('equal')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', ncol=1, fontsize=9, bbox_to_anchor=(.5, .62))
    fig.suptitle('Reviewed +Y cup mesh: ideal bearing, motor-bolt and distal load patches', fontsize=12)
    fig.savefig(HERE / 'boundary-and-source-mesh.png', dpi=190)
    plt.close(fig)
    report = {name:{'triangles':int(mask.sum()),'area_mm2':float(area[mask].sum()),
                    'nodes':int(len(np.unique(tri[mask])))} for name,mask in groups.items()}
    if min(item['triangles'] for item in report.values()) < 10:
        raise ValueError('boundary patch undersampled')
    (HERE / 'boundary-patches.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
