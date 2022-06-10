import { StyleRules, withStyles } from "@material-ui/core/styles"
import createStyles from "@material-ui/core/styles/createStyles"
import * as React from "react"
import { connect } from "react-redux"
import * as THREE from "three"
import { Sensor } from "../common/sensor"

import Session from "../common/session"
import { getMainSensor } from "../common/util"
import { DataType, ItemTypeName } from "../const/common"
import {
  getMinSensorIds,
  isCurrentFrameLoaded,
  isCurrentItemLoaded
} from "../functional/state_util"
import {
  ColorSchemeType,
  PointCloudViewerConfigType,
  State
} from "../types/state"
import {
  DrawableCanvas,
  DrawableProps,
  mapStateToDrawableProps
} from "./viewer"

const styles = (): StyleRules<"point_cloud_canvas", {}> =>
  createStyles({
    point_cloud_canvas: {
      position: "absolute",
      height: "100%",
      width: "100%"
    }
  })

interface ClassType {
  /** CSS canvas name */
  point_cloud_canvas: string
}

interface Props extends DrawableProps {
  /** CSS class */
  classes: ClassType
  /** container */
  display: HTMLDivElement | null
  /** viewer id */
  id: number
  /** camera */
  camera: THREE.PerspectiveCamera
}

const vertexShader = `
    varying vec3 worldPosition;
    attribute vec3 color;
    varying vec3 pointColor;
    void main() {
      vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
      gl_PointSize = 3.0;
      gl_Position = projectionMatrix * mvPosition;
      worldPosition = position;
      pointColor = color;
    }
  `
const fragmentShader = `
    varying vec3 worldPosition;

    uniform vec3 low;
    uniform vec3 high;

    uniform mat4 toSelectionFrame;
    uniform vec3 selectionSize;

    varying vec3 pointColor;

    bool pointInSelection(vec3 point) {
      vec4 testPoint = abs(toSelectionFrame * vec4(point.xyz, 1.0));
      vec3 halfSize = selectionSize / 2.;
      return testPoint.x < halfSize.x &&
        testPoint.y < halfSize.y &&
        testPoint.z < halfSize.z;
    }

    bool pointInNearby(vec3 point) {
      vec4 testPoint = abs(toSelectionFrame * vec4(point.xyz, 1.0));
      vec3 halfSize = selectionSize / 2.;
      float expandX = min(halfSize.x, 0.5);
      float expandY = min(halfSize.y, 0.5);
      float expandZ = min(halfSize.z, 0.5);
      return (testPoint.x < halfSize.x + expandX &&
              testPoint.y < halfSize.y + expandY &&
              testPoint.z < halfSize.z + expandZ) &&
             (testPoint.x > halfSize.x ||
              testPoint.y > halfSize.y ||
              testPoint.z > halfSize.z);
    }

    void main() {
      float alpha = 0.5;
      vec3 color = pointColor;
      if (
        selectionSize.x * selectionSize.y * selectionSize.z > 1e-4
      ) {
        if (pointInSelection(worldPosition)) {
          alpha = 1.0;
          color.x *= 2.0;
          color.yz *= 0.5;
        } else if (pointInNearby(worldPosition)) {
          alpha = 1.0;
          color.x = 0.0;
          color.y = 256.0;
          color.z = 0.0;
        };
      } else {
        alpha = 1.0;
      }
      gl_FragColor = vec4(color, alpha);
    }
  `

/**
 * Canvas Viewer
 */
class PointCloudCanvas extends DrawableCanvas<Props> {
  /** Container */
  private display: HTMLDivElement | null
  /** Canvas to draw on */
  private canvas: HTMLCanvasElement | null
  /** ThreeJS Renderer */
  private renderer?: THREE.WebGLRenderer
  /** ThreeJS Scene object */
  private scene: THREE.Scene
  /** ThreeJS Camera */
  private readonly camera: THREE.PerspectiveCamera
  /** ThreeJS sphere mesh for indicating camera target location */
  private readonly target: THREE.AxesHelper
  /** Current point cloud for rendering */
  private readonly pointCloud: THREE.Points
  /** drawable callback */
  private readonly _drawableUpdateCallback: () => void
  /** have points been set or transformed */
  private _pointsUpdated: boolean
  /** currently selected item */
  private _currentItem: number
  /** context of image canvas */
  private _hiddenContext: CanvasRenderingContext2D | null
  /** canvas for drawing image & getting colors */
  private _hiddenCanvas: HTMLCanvasElement
  /** point cloud color scheme */
  private _colorScheme: ColorSchemeType | null
  /** image data */
  private _imageData: Uint8ClampedArray | null

  /**
   * Constructor, ons subscription to store
   *
   * @param {Object} props: react props
   * @param props
   */
  constructor(props: Readonly<Props>) {
    super(props)
    this.scene = new THREE.Scene()
    this.camera = props.camera
    this.target = new THREE.AxesHelper(0.5)
    this.scene.add(this.target)
    this._currentItem = 0
    this._pointsUpdated = false
    this._currentItem = 0
    this._hiddenContext = null
    this._hiddenCanvas = document.createElement("canvas")
    this._colorScheme = null
    this._imageData = null

    this.canvas = null
    this.display = null

    this._drawableUpdateCallback = this.renderThree.bind(this)

    const material = new THREE.ShaderMaterial({
      uniforms: {
        low: {
          value: new THREE.Color(0x0000ff)
        },
        high: {
          value: new THREE.Color(0xffff00)
        },
        toSelectionFrame: {
          value: new THREE.Matrix4()
        },
        selectionSize: {
          value: new THREE.Vector3()
        }
      },
      vertexShader,
      fragmentShader,
      transparent: true
    })

    this.pointCloud = new THREE.Points(new THREE.BufferGeometry(), material)
  }

  /** mount callback */
  public componentDidMount(): void {
    super.componentDidMount()
    Session.label3dList.subscribe(this._drawableUpdateCallback)
  }

  /** mount callback */
  public componentWillUnmount(): void {
    super.componentWillUnmount()
    Session.label3dList.unsubscribe(this._drawableUpdateCallback)
  }

  /**
   * Render function
   *
   * @return {React.Fragment} React fragment
   */
  public render(): JSX.Element {
    const { classes } = this.props

    let canvas = (
      <canvas
        key={`point-cloud-canvas-${this.props.id}`}
        className={classes.point_cloud_canvas}
        ref={(ref) => {
          this.initializeRefs(ref)
        }}
      />
    )

    if (this.display !== null) {
      const displayRect = this.display.getBoundingClientRect()
      canvas = React.cloneElement(canvas, {
        height: displayRect.height,
        width: displayRect.width
      })
    }

    this.redraw()

    return canvas
  }

  /**
   * Handles canvas redraw
   *
   * @return {boolean}
   */
  public redraw(): boolean {
    const state = this.state
    if (isCurrentItemLoaded(state) && this.canvas !== null) {
      const sensor = this.state.user.viewerConfigs[this.props.id].sensor
      if (isCurrentFrameLoaded(this.state, sensor)) {
        this.updateRenderer()
        this.renderThree()
      } else {
        this.renderer?.clear()
      }
    }
    return true
  }

  /**
   * Override method
   *
   * @param _state
   * @param state
   */
  protected updateState(state: State): void {
    if (this.display !== this.props.display) {
      this.display = this.props.display
      this.forceUpdate()
    }
    const select = state.user.select
    const item = select.item
    const sensor = this.state.user.viewerConfigs[this.props.id].sensor
    if (item !== this._currentItem) {
      this._currentItem = item
      this._pointsUpdated = false
    }

    const config = state.user.viewerConfigs[
      this.props.id
    ] as PointCloudViewerConfigType
    this.target.position.set(config.target.x, config.target.y, config.target.z)

    if (
      Session.pointClouds[item][sensor] !== undefined &&
      !this._pointsUpdated
    ) {
      const mainSensor = getMainSensor(this.state)
      const geometry = Session.pointClouds[item][mainSensor.id]
      this.pointCloud.geometry.copy(geometry)
      this.pointCloud.layers.enableAll()
      this._pointsUpdated = true

      const itemType = this.state.task.config.itemType
      if (itemType === ItemTypeName.IMAGE) {
        const minSensorIds = getMinSensorIds(this.state)
        const imageSensorId = minSensorIds[DataType.IMAGE]
        const image = Session.images[item][imageSensorId]
        if (this._hiddenContext === null) {
          this._hiddenCanvas.width = image.width
          this._hiddenCanvas.height = image.height
          this._hiddenContext = this._hiddenCanvas.getContext("2d")
        }
        if (this._hiddenContext !== null) {
          this._hiddenContext.drawImage(image, 0, 0)
          this._imageData = this._hiddenContext.getImageData(
            0,
            0,
            this._hiddenCanvas.width,
            this._hiddenCanvas.height
          ).data
        }
      }
      this.updatePointCloudColors()
    }
    if (this._pointsUpdated && this._colorScheme !== config.colorScheme) {
      this._colorScheme = config.colorScheme
      this.updatePointCloudColors()
    }
  }

  /**
   * Update point cloud color scheme
   */
  private updatePointCloudColors(): void {
    if (this._colorScheme === null) {
      return
    }
    let colors: number[] = []
    switch (this._colorScheme) {
      case ColorSchemeType.IMAGE:
        colors = this.getImageColors()
        break
      case ColorSchemeType.DEPTH:
        colors = this.getDepthColors()
        break
      case ColorSchemeType.HEIGHT:
        colors = this.getHeightColors()
    }
    const geometry = this.pointCloud.geometry
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3))
  }

  /**
   * Set color for points based on depth
   */
  private getDepthColors(): number[] {
    const geometry = this.pointCloud.geometry
    const points = Array.from(geometry.getAttribute("position").array)
    const depths: number[] = []
    for (let i = 0; i < points.length; i += 3) {
      const point = new THREE.Vector3(points[i], points[i + 1], points[i + 2])
      const depth = point.length()
      depths.push(depth)
    }
    const maxDepth = Math.min(Math.max.apply(Math, depths), 60)
    const minDepth = Math.min.apply(Math, depths)
    const colors: number[] = []
    for (let i = 0; i < depths.length; i += 1) {
      const depth = depths[i]
      const hue = Math.max(0, 0.66 - (depth - minDepth) / (maxDepth - minDepth))
      const color = new THREE.Color().setHSL(hue, 1.0, 0.5)
      colors.push(color.r, color.g, color.b)
    }
    return colors
  }

  /**
   * Set color for points based on height
   */
  private getHeightColors(): number[] {
    const quantile = (arr: number[], q: number): number => {
      const sorted = [...arr].sort((a, b) => a - b)
      const pos = (sorted.length - 1) * q
      const base = Math.floor(pos)
      const rest = pos - base
      if (sorted[base + 1] !== undefined) {
        return sorted[base] + rest * (sorted[base + 1] - sorted[base])
      } else {
        return sorted[base]
      }
    }
    const mainSensor = getMainSensor(this.state)
    const up = mainSensor.up.clone().normalize()
    const geometry = this.pointCloud.geometry
    const points = Array.from(geometry.getAttribute("position").array)
    const heights: number[] = []
    for (let i = 0; i < points.length; i += 3) {
      const point = new THREE.Vector3(points[i], points[i + 1], points[i + 2])
      const height = point.dot(up)
      heights.push(height)
    }
    const maxHeight = quantile(heights, 0.95)
    const minHeight = quantile(heights, 0.05)
    const colors: number[] = []
    const lowColor = new THREE.Color(0x0000ff)
    const highColor = new THREE.Color(0xffff00)
    for (let i = 0; i < heights.length; i += 1) {
      const height = heights[i]
      const fraction = Math.min(
        1.0,
        Math.max(0.0, (height - minHeight) / (maxHeight - minHeight))
      )
      const color = lowColor
        .clone()
        .multiplyScalar(1.0 - fraction)
        .add(highColor.clone().multiplyScalar(fraction))
      colors.push(color.r, color.g, color.b)
    }
    return colors
  }

  /**
   * Set color for points based on image
   */
  private getImageColors(): number[] {
    const minSensorIds = getMinSensorIds(this.state)
    const sensorId = minSensorIds[DataType.IMAGE]
    const sensorType = this.state.task.sensors[sensorId]
    const item = this.state.user.select.item
    const image = Session.images[item][sensorId]
    const sensor = Sensor.fromSensorType(sensorType, image)
    if (sensor.hasIntrinsics() && this._imageData !== null) {
      const geometry = this.pointCloud.geometry
      const points = Array.from(geometry.getAttribute("position").array)
      const colors: number[] = []
      for (let i = 0; i < points.length; i += 3) {
        const point = new THREE.Vector3(points[i], points[i + 1], points[i + 2])
        const pixel = sensor.project(point)
        pixel.divideScalar(pixel.z)
        let color = new THREE.Vector3(0, 0, 1)
        if (
          point.z > 0 &&
          this._hiddenContext !== null &&
          pixel.x < 1 &&
          pixel.x > 0 &&
          pixel.y < 1 &&
          pixel.y > 0
        ) {
          const x = Math.floor(pixel.x * image.width)
          const y = Math.floor(pixel.y * image.height)
          const imageIndex = (y * this._hiddenCanvas.width + x) * 4
          const rgb = this._imageData.slice(imageIndex, imageIndex + 3)
          color = new THREE.Vector3(rgb[0], rgb[1], rgb[2])
          color.divideScalar(255.0)
        }
        colors.push(color.x, color.y, color.z)
      }
      return colors
    }
    return []
  }

  /**
   * Render ThreeJS Scene
   */
  private renderThree(): void {
    if (this.renderer !== undefined) {
      this.scene.children = []
      this.scene.add(this.pointCloud)
      this.scene.add(this.target)

      const selectionTransform = new THREE.Matrix4()
      const selectionSize = new THREE.Vector3()
      if (Session.label3dList.selectedLabel !== null) {
        const label = Session.label3dList.selectedLabel
        const selectionToWorld = new THREE.Matrix4()
        selectionToWorld.makeRotationFromQuaternion(label.orientation)
        selectionToWorld.setPosition(label.center)
        selectionTransform.copy(selectionToWorld).invert()
        selectionSize.copy(label.size)
      }

      const material = this.pointCloud.material as THREE.ShaderMaterial
      material.uniforms.toSelectionFrame.value = selectionTransform
      material.uniforms.selectionSize.value = selectionSize

      this.renderer.render(this.scene, this.camera)
    }
  }

  /**
   * Set references to div elements and try to initialize renderer
   *
   * @param {HTMLDivElement} component
   * @param {string} componentType
   */
  private initializeRefs(component: HTMLCanvasElement | null): void {
    if (component === null) {
      return
    }

    if (component.nodeName === "CANVAS") {
      if (this.canvas !== component) {
        this.canvas = component
        const rendererParams = { canvas: this.canvas, alpha: true }
        this.renderer = new THREE.WebGLRenderer(rendererParams)
        this.forceUpdate()
      }

      if (this.display !== null) {
        this.canvas.removeAttribute("style")
        const displayRect = this.display.getBoundingClientRect()
        this.canvas.width = displayRect.width
        this.canvas.height = displayRect.height
      }

      if (isCurrentItemLoaded(this.state)) {
        this.updateRenderer()
      }
    }
  }

  /**
   * Update rendering constants
   */
  private updateRenderer(): void {
    if (this.canvas !== null && this.renderer !== undefined) {
      this.renderer.setSize(this.canvas.width, this.canvas.height)
    }
  }
}

const styledCanvas = withStyles(styles, { withTheme: true })(PointCloudCanvas)
export default connect(mapStateToDrawableProps)(styledCanvas)
