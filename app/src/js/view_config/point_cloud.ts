import * as THREE from 'three'
import { PointCloudViewerConfigType } from '../functional/types'

/**
 * Update ThreeJS rendering objects with viewer config params
 * @param canvas
 * @param config
 * @param renderer
 * @param camera
 * @param target
 */
export function updateThreeCameraAndRenderer (
  canvas: HTMLCanvasElement,
  config: PointCloudViewerConfigType,
  renderer: THREE.Renderer,
  camera: THREE.PerspectiveCamera,
  target: THREE.Object3D
) {
  target.position.x = config.target.x
  target.position.y = config.target.y
  target.position.z = config.target.z

  if (canvas) {
    camera.aspect = canvas.offsetWidth /
      canvas.offsetHeight
    camera.updateProjectionMatrix()
  }

  camera.up.x = config.verticalAxis.x
  camera.up.y = config.verticalAxis.y
  camera.up.z = config.verticalAxis.z
  camera.position.x = config.position.x
  camera.position.y = config.position.y
  camera.position.z = config.position.z
  camera.lookAt(target.position)

  if (renderer && canvas) {
    renderer.setSize(canvas.width,
      canvas.height)
  }
}

/**
 * Normalize mouse coordinates
 * @param {number} mX: Mouse x-coord
 * @param {number} mY: Mouse y-coord
 * @return {Array<number>}
 */
export function convertMouseToNDC (
  mX: number,
  mY: number,
  canvas: HTMLCanvasElement
): number[] {
  let x = mX / canvas.offsetWidth
  let y = mY / canvas.offsetHeight
  x = 2 * x - 1
  y = -2 * y + 1

  return [x, y]
}

/**
 * Get projection from mouse into scene
 * @param x
 * @param y
 * @param camera
 */
export function projectionFromNDC (
  x: number, y: number, camera: THREE.Camera
): THREE.Ray {
  const direction = new THREE.Vector3(x, y, -1)

  direction.unproject(camera)
  direction.sub(camera.position)
  direction.normalize()

  const projection = new THREE.Ray(camera.position, direction)

  return projection
}
