import { StyleRules, withStyles } from "@material-ui/core/styles"
import createStyles from "@material-ui/core/styles/createStyles"
import * as React from "react"
import { connect } from "react-redux"
import * as THREE from "three"

import Session from "../common/session"
import {
  isCurrentFrameLoaded,
  isCurrentItemLoaded
} from "../functional/state_util"
import { PointCloudViewerConfigType, State } from "../types/state"
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
    void main() {
      vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
      gl_PointSize = 0.1 * ( 300.0 / -mvPosition.z );
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

    vec3 getHeatMapColor(float height) {
      float val = min(1.0, max(0.0, (height + 3.0) / 6.0));
      return (1.0 - val) * low + val * high;
    }

    bool pointInSelection(vec3 point) {
      vec4 testPoint = abs(toSelectionFrame * vec4(point.xyz, 1.0));
      vec3 halfSize = selectionSize / 2.;
      return testPoint.x < halfSize.x &&
        testPoint.y < halfSize.y &&
        testPoint.z < halfSize.z;
    }

    void main() {
      float alpha = 0.5;
      vec3 color = getHeatMapColor(worldPosition.z);
      if (
        selectionSize.x * selectionSize.y * selectionSize.z > 1e-4
      ) {
        if (pointInSelection(worldPosition)) {
          alpha = 1.0;
          color.x *= 2.0;
          color.yz *= 0.5;
        }
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
  private pointCloud: THREE.Points
  /** drawable callback */
  private readonly _drawableUpdateCallback: () => void

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

    this.pointCloud = new THREE.Points(undefined, material)
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

    this.pointCloud.geometry = Session.pointClouds[item][sensor]
    this.pointCloud.layers.enableAll()

    const config = state.user.viewerConfigs[
      this.props.id
    ] as PointCloudViewerConfigType
    this.target.position.set(config.target.x, config.target.y, config.target.z)
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
        selectionTransform.getInverse(selectionToWorld)
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
