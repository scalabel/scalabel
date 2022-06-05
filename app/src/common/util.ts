import { ADD_LABELS } from "../const/action"
import {
  ItemTypeName,
  LabelTypeName,
  ViewerConfigTypeName
} from "../const/common"
import { getMinSensorIds } from "../functional/state_util"
import { ActionPacketType } from "../types/message"
import { State } from "../types/state"
import { Sensor } from "./sensor"
import * as THREE from "three"

/**
 * Handle invalid page request
 */
export function handleInvalidPage(): void {
  window.location.replace(window.location.origin)
}

/**
 * Get whether tracking is on
 * Also get the new item type
 *
 * @param itemType
 */
export function getTracking(itemType: string): [string, boolean] {
  switch (itemType) {
    case ItemTypeName.VIDEO:
      return [ItemTypeName.IMAGE, true]
    case ItemTypeName.POINT_CLOUD_TRACKING:
      return [ItemTypeName.POINT_CLOUD, true]
    case ItemTypeName.FUSION:
      return [ItemTypeName.FUSION, true]
    default:
      return [itemType, false]
  }
}

/**
 * Return viewer type required for given sensor and label types
 *
 * @param sensorType
 * @param labelTypes
 */
export function getViewerType(
  sensorType: ViewerConfigTypeName,
  labelTypes: LabelTypeName[]
): ViewerConfigTypeName {
  if (
    sensorType === ViewerConfigTypeName.IMAGE &&
    labelTypes.includes(LabelTypeName.BOX_3D)
  ) {
    return ViewerConfigTypeName.IMAGE_3D
  }
  return sensorType
}

/**
 * Create the link to the labeling instructions
 *
 * @param pageName
 */
function makeInstructionUrl(pageName: string): string {
  return `https://doc.scalabel.ai/instructions/${pageName}.html`
}

/**
 * Select the correct instruction url for the given label type
 *
 * @param labelType
 */
export function getInstructionUrl(labelType: string): string {
  switch (labelType) {
    case LabelTypeName.BOX_2D: {
      return makeInstructionUrl("bbox")
    }
    case LabelTypeName.POLYGON_2D:
    case LabelTypeName.POLYLINE_2D: {
      return makeInstructionUrl("segmentation")
    }
    default: {
      return ""
    }
  }
}

/**
 * Select the correct page title for given label type
 *
 * @param labelType
 * @param itemType
 */
export function getPageTitle(labelType: string, itemType: string): string {
  const [, tracking] = getTracking(itemType)

  let title: string
  switch (labelType) {
    case LabelTypeName.TAG:
      title = "Image Tagging"
      break
    case LabelTypeName.BOX_2D:
      title = "2D Bounding Box"
      break
    case LabelTypeName.POLYGON_2D:
      title = "2D Segmentation"
      break
    case LabelTypeName.POLYLINE_2D:
      title = "2D Lane"
      break
    case LabelTypeName.BOX_3D:
      title = "3D Bounding Box"
      break
    default:
      title = ""
      break
  }
  if (tracking) {
    title = `${title} Tracking`
  }
  return title
}

/**
 * Converts index into a filename of size 6 with
 * trailing zeroes
 *
 * @param index
 */
export function index2str(index: number): string {
  return index.toString().padStart(6, "0")
}

/**
 * Checks if the action packet contains
 * any actions that would trigger a model query
 *
 * @param actionPacket
 * @param bots
 */
export function doesPacketTriggerModel(
  actionPacket: ActionPacketType,
  bots: boolean
): boolean {
  if (!bots) {
    return false
  }
  for (const action of actionPacket.actions) {
    if (action.type === ADD_LABELS) {
      return true
    }
  }
  return false
}

/**
 * Transform point cloud with sensor extrinsics
 *
 * @param fromGeometry
 * @param fromSensorId
 * @param state
 */
export function transformPointCloud(
  fromGeometry: THREE.BufferGeometry,
  fromSensorId: number,
  state: State
): THREE.BufferGeometry {
  const sensorType = state.task.sensors[fromSensorId]
  const sensor = Sensor.fromSensorType(sensorType)
  const mainSensor = getMainSensor(state)
  const geometry = fromGeometry.clone()
  const points = Array.from(geometry.getAttribute("position").array)
  const newPoints: number[] = []
  for (let i = 0; i < points.length; i += 3) {
    const point = new THREE.Vector3(points[i], points[i + 1], points[i + 2])
    const newPoint = mainSensor.inverseTransform(sensor.transform(point))
    newPoints.push(newPoint.x, newPoint.y, newPoint.z)
  }
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(newPoints, 3)
  )
  return geometry
}

/**
 * Get main sensor object from state
 *
 * @param state
 */
export function getMainSensor(state: State): Sensor {
  const itemType = state.task.config.itemType
  const minSensorIds = getMinSensorIds(state)
  const mainSensor = state.task.sensors[minSensorIds[itemType]]
  const sensor = Sensor.fromSensorType(mainSensor)
  return sensor
}

/** Ground plane estimation using RANSAC
 *
 * @param points
 */
export function estimateGroundPlane(points: THREE.Vector3[]): THREE.Plane {
  let bestPlane = new THREE.Plane()
  let maxNumPoints = 0
  const itMax = Math.ceil(points.length / 40)
  const threshold = 0.01
  for (let i = 0; i < itMax; i++) {
    let p1 = points[getRandomInt(points.length)]
    let p2 = points[getRandomInt(points.length)]
    let p3 = points[getRandomInt(points.length)]
    // avoid sampling same points
    while (
      new THREE.Vector3().subVectors(p1, p2).length() < threshold ||
      new THREE.Vector3().subVectors(p2, p3).length() < threshold ||
      new THREE.Vector3().subVectors(p3, p1).length() < threshold
    ) {
      p1 = points[getRandomInt(points.length)]
      p2 = points[getRandomInt(points.length)]
      p3 = points[getRandomInt(points.length)]
    }
    const plane = new THREE.Plane().setFromCoplanarPoints(p1, p2, p3)
    let numPoints = 0
    for (let p = 0; p < points.length; p++) {
      if (Math.abs(plane.distanceToPoint(points[p])) < threshold) {
        numPoints += 1
      }
    }
    if (numPoints > maxNumPoints) {
      bestPlane = plane
      maxNumPoints = numPoints
    }
  }
  return bestPlane
}

/**
 * Get points from raw vertices
 *
 * @param vertices
 */
export function getPointsFromVertices(vertices: number[]): THREE.Vector3[] {
  const points: THREE.Vector3[] = []
  for (let i = 0; i < vertices.length; i += 3) {
    points.push(
      new THREE.Vector3(vertices[i], vertices[i + 1], vertices[i + 2])
    )
  }
  return points
}

/** Random integer generator
 *
 * @param max
 */
function getRandomInt(max: number): number {
  return Math.floor(Math.random() * max)
}

/**
 * Calculate rotation for estimated plane
 *
 * @param up
 * @param left
 * @param estimatedNormal
 */
export function calculatePlaneRotation(
  up: THREE.Vector3,
  left: THREE.Vector3,
  estimatedNormal: THREE.Vector3
): THREE.Vector3 {
  const rotation = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 0, 1),
    estimatedNormal
  )
  // Normal should face roughly up
  const angleToNormal = up.angleTo(estimatedNormal)
  if (angleToNormal > Math.PI / 2) {
    const flipRotation = new THREE.Euler().setFromVector3(
      left.clone().normalize().multiplyScalar(Math.PI)
    )
    const flipQuarternion = new THREE.Quaternion().setFromEuler(flipRotation)
    rotation.multiply(flipQuarternion)
  }
  const rotationEuler = new THREE.Euler().setFromQuaternion(rotation)
  // No rotation around vertical axis
  rotationEuler.z = 0
  return rotationEuler.toVector3()
}

/**
 * Calcuate estimated plane center.
 * Assume the plane center is directly below target center.
 *
 * @param plane
 * @param target
 * @param down
 */
export function calculatePlaneCenter(
  plane: THREE.Plane,
  target: THREE.Vector3,
  down: THREE.Vector3
): THREE.Vector3 {
  const ray = new THREE.Ray(target, down)
  const center = new THREE.Vector3()
  ray.intersectPlane(plane, center)
  return center
}

/**
 * Get the closest point to the ray interception with the pointCloud
 *
 * @param pointCloud
 * @param ray
 */
export function getClosestPoint(
  pointCloud: THREE.Points,
  ray: THREE.Ray
): THREE.Vector3 | null {
  const raycaster = new THREE.Raycaster()
  raycaster.set(ray.origin, ray.direction)

  const points = Array.from(pointCloud.geometry.getAttribute("position").array)
  let minDist = Infinity
  let closestPoint = null
  for (let i = 0; i < points.length; i += 3) {
    const point = new THREE.Vector3(points[i], points[i + 1], points[i + 2])
    const dist = ray.distanceToPoint(point)
    if (dist < minDist) {
      minDist = dist
      closestPoint = point
    }
  }
  return closestPoint
}
