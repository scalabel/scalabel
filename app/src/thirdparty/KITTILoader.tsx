import * as THREE from "three"

/** KITTI loader for velodyne data files */
export class KITTILoader {
  /**
   * load KITTI bin files as pointclouds
   *
   * @param url
   */
  public load(url: string): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry()
    const req = new XMLHttpRequest()
    req.open("GET", url)
    req.responseType = "arraybuffer" // the important part
    req.onreadystatechange = function () {
      if (req.readyState === 4) {
        const vertices = new Float32Array(req.response)
        geometry.setAttribute(
          "position",
          new THREE.BufferAttribute(vertices, 4)
        )
      }
    }
    req.send()
    console.log(geometry.attributes)
    return geometry
  }
}
