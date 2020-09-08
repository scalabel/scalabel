import * as THREE from "three"

import { Vector3D } from "../math/vector3d"
import { PointCloudViewerConfigType } from "../types/state"

/**
 * Update ThreeJS rendering objects with viewer config params
 *
 * @param canvas
 * @param config
 * @param renderer
 * @param camera
 * @param target
 */
export function updateThreeCameraAndRenderer(
  config: PointCloudViewerConfigType,
  camera: THREE.Camera,
  canvas?: HTMLCanvasElement,
  renderer?: THREE.Renderer
): void {
  if (canvas !== undefined) {
    ;(camera as THREE.PerspectiveCamera).aspect = canvas.width / canvas.height
    ;(camera as THREE.PerspectiveCamera).updateProjectionMatrix()
  }

  camera.up.x = config.verticalAxis.x
  camera.up.y = config.verticalAxis.y
  camera.up.z = config.verticalAxis.z
  camera.position.x = config.position.x
  camera.position.y = config.position.y
  camera.position.z = config.position.z
  camera.lookAt(new Vector3D().fromState(config.target).toThree())

  if (renderer !== undefined && canvas !== undefined) {
    renderer.setSize(canvas.width, canvas.height)
  }
}

/**
 * Normalize mouse coordinates
 *
 * @param {number} mX: Mouse x-coord
 * @param {number} mY: Mouse y-coord
 * @param mX
 * @param mY
 * @param canvas
 * @returns {Array<number>}
 */
export function convertMouseToNDC(
  mX: number,
  mY: number,
  canvas: HTMLCanvasElement
): number[] {
  let x = mX / canvas.width
  let y = mY / canvas.height
  x = 2 * x - 1
  y = -2 * y + 1

  return [x, y]
}

/**
 * Get projection from mouse into scene
 *
 * @param x
 * @param y
 * @param camera
 */
export function projectionFromNDC(
  x: number,
  y: number,
  camera: THREE.Camera
): THREE.Ray {
  const direction = new THREE.Vector3(x, y, -1)

  direction.unproject(camera)
  direction.sub(camera.position)
  direction.normalize()

  const projection = new THREE.Ray(camera.position, direction)

  return projection
}
