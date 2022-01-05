import { StyleRules, withStyles } from "@material-ui/core/styles"
import createStyles from "@material-ui/core/styles/createStyles"
import { MAX_SCALE, MIN_SCALE, updateCanvasScale } from "../view_config/image"
import * as React from "react"
import { connect } from "react-redux"
import * as THREE from "three"

import Session from "../common/session"
import {
  isCurrentFrameLoaded,
  isCurrentItemLoaded
} from "../functional/state_util"
import { Image3DViewerConfigType, State } from "../types/state"
import {
  DrawableCanvas,
  DrawableProps,
  mapStateToDrawableProps
} from "./viewer"
import { ViewerConfigTypeName } from "../const/common"

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
  /** sensor id */
  sensor: number
  /** viewer id */
  id: number
  /** camera */
  camera: THREE.Camera
}

const vertexShader = `
    varying vec3 worldPosition;
    attribute float size;
    void main() {
      vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
      gl_PointSize = size;
      gl_Position = projectionMatrix * mvPosition;
      worldPosition = position;
    }
  `
const fragmentShader = `
    varying vec3 worldPosition;

    uniform vec3 low;
    uniform vec3 high;

    uniform mat4 toSelectionFrame;
    uniform vec3 selectionSize;

    vec3 hsv2rgb(vec3 c)
    {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    vec3 getHeatMapColor(float distance, float max_val) {
      float val = min(max_val, max(0.0, distance)) / max_val;
      vec3 hsv_val = vec3(val, 1.0, 1.0);
      // float h = (1.0 - val) * low + val * high;
      vec3 rgb_val = hsv2rgb(hsv_val);
      return rgb_val;
    }

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
      float distance = sqrt(
        pow(worldPosition.x, 2.0)+
        pow(worldPosition.y, 2.0)+
        pow(worldPosition.z, 2.0));
      vec3 color = getHeatMapColor(distance, 23.0);
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
class PointCloudOverlayCanvas extends DrawableCanvas<Props> {
  /** Container */
  private display: HTMLDivElement | null
  /** Canvas to draw on */
  private canvas: HTMLCanvasElement | null
  /** ThreeJS Renderer */
  private renderer?: THREE.WebGLRenderer
  /** ThreeJS Scene object */
  private scene: THREE.Scene
  /** ThreeJS Camera */
  private readonly camera: THREE.Camera
  /** Current point cloud for rendering */
  private readonly pointCloud: THREE.Points
  /** have points been transformed */
  private transformedPoints: boolean
  /** have points been cloned */
  private pointsCloned: boolean

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
    this.transformedPoints = false
    this.pointsCloned = false

    this.canvas = null
    this.display = null

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

  /**
   * Render function
   *
   * @return {React.Fragment} React fragment
   */
  public render(): JSX.Element {
    const { classes } = this.props

    let canvas = (
      <canvas
        key={`point-cloud--overlay-canvas-${this.props.sensor}`}
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
      const sensor = this.props.sensor
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
   * @param state
   */
  protected updateState(state: State): void {
    if (this.display !== this.props.display) {
      this.display = this.props.display
      this.forceUpdate()
    }
    const item = state.user.select.item
    const sensor = this.props.sensor

    if (Session.pointClouds[item][sensor] !== undefined && !this.pointsCloned) {
      this.pointCloud.geometry = Session.pointClouds[item][sensor].clone()
      this.pointCloud.layers.enableAll()
      this.transformPoints()
      this.pointsCloned = true
    }
  }

  /**
   * Transform points with sensor extrinsics
   */
  private transformPoints(): void {
    const sensor = this.state.task.sensors[this.props.sensor]
    const extrinsics = sensor.extrinsics
    if (
      !this.transformedPoints &&
      extrinsics !== undefined &&
      this.pointCloud.geometry !== undefined
    ) {
      const geometry = this.pointCloud.geometry
      const rawPoints = Array.from(geometry.getAttribute("position").array)
      const points: THREE.Vector3[] = []
      for (let i = 0; i < rawPoints.length; i += 3) {
        points.push(
          new THREE.Vector3(rawPoints[i], rawPoints[i + 1], rawPoints[i + 2])
        )
      }
      const rotation4 = new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(
          -extrinsics.rotation.x,
          -extrinsics.rotation.y,
          -extrinsics.rotation.z
        )
      )
      const position = new THREE.Vector3(
        extrinsics.translation.x,
        extrinsics.translation.y,
        extrinsics.translation.z
      )
      position.applyMatrix4(rotation4).multiplyScalar(-1)
      rotation4.setPosition(position)
      const newRawPoints: number[] = []
      const sizes = new Float32Array(points.length)
      for (let i = 0; i < points.length; i += 1) {
        const point = points[i]
        point.applyMatrix4(rotation4)
        newRawPoints.push(point.x)
        newRawPoints.push(point.y)
        newRawPoints.push(point.z)
        sizes[i] = 1.5
      }
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(newRawPoints, 3)
      )
      this.transformedPoints = true
      geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1))
      geometry.attributes.size.needsUpdate = true
    }
  }

  /**
   * Render ThreeJS Scene
   */
  private renderThree(): void {
    if (this.renderer !== undefined && this.pointCloud.geometry !== undefined) {
      console.log("rendering!")
      this.scene.children = []
      this.scene.add(this.pointCloud)

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
    const viewerConfig = this.state.user.viewerConfigs[this.props.id]
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

      if (
        this.canvas !== null &&
        this.display !== null &&
        viewerConfig.type === ViewerConfigTypeName.IMAGE_3D
      ) {
        const img3dConfig = viewerConfig as Image3DViewerConfigType
        if (
          img3dConfig.viewScale >= MIN_SCALE &&
          img3dConfig.viewScale < MAX_SCALE
        ) {
          updateCanvasScale(
            this.state,
            this.display,
            this.canvas,
            null,
            img3dConfig,
            img3dConfig.viewScale,
            false
          )
        }
      } else if (this.display !== null) {
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

const styledCanvas = withStyles(styles, { withTheme: true })(
  PointCloudOverlayCanvas
)
export default connect(mapStateToDrawableProps)(styledCanvas)
