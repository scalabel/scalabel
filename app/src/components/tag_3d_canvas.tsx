import { StyleRules, withStyles } from "@material-ui/core/styles"
import createStyles from "@material-ui/core/styles/createStyles"
import * as React from "react"
import { connect } from "react-redux"
import * as THREE from "three"

import Session from "../common/session"
import { ViewerConfigTypeName } from "../const/common"
import { isCurrentFrameLoaded } from "../functional/state_util"
import { makeTaskConfig } from "../functional/states"
import { ConfigType, Image3DViewerConfigType, State } from "../types/state"
import { MAX_SCALE, MIN_SCALE, updateCanvasScale } from "../view_config/image"
import {
  DrawableCanvas,
  DrawableProps,
  mapStateToDrawableProps
} from "./viewer"

const styles = (): StyleRules<"tag3d_canvas", {}> =>
  createStyles({
    tag3d_canvas: {
      position: "absolute",
      height: "100%",
      width: "100%",
      "pointer-events": "none"
    }
  })

interface ClassType {
  /** CSS canvas name */
  tag3d_canvas: string
}

interface Props extends DrawableProps {
  /** CSS class */
  classes: ClassType
  /** container */
  display: HTMLDivElement | null
  /** viewer id */
  id: number
  /** camera */
  camera: THREE.Camera
}

/**
 * Canvas Viewer
 */
export class Tag3dCanvas extends DrawableCanvas<Props> {
  /** Canvas to draw on */
  private canvas: HTMLCanvasElement | null
  /** Canvas context */
  private _context: CanvasRenderingContext2D | null
  /** Container */
  private display: HTMLDivElement | null
  /** Current scale */
  private scale: number
  /** ThreeJS Camera */
  private readonly camera: THREE.Camera
  /** Flag set if data is 2d */
  private data2d: boolean
  /** task config */
  private _config: ConfigType

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
    this.camera = props.camera

    this.display = null
    this.canvas = null
    this._context = null
    this.scale = 1
    this.data2d = false
    this._config = makeTaskConfig()

    this._drawableUpdateCallback = this.redraw.bind(this)
  }

  /**
   * Mount callback
   */
  public componentDidMount(): void {
    super.componentDidMount()
    Session.label3dList.subscribe(this._drawableUpdateCallback)
  }

  /**
   * Unmount callback
   */
  public componentWillUnmount(): void {
    super.componentWillUnmount()
    Session.label3dList.unsubscribe(this._drawableUpdateCallback)
  }

  /**
   * Render function
   */
  public render(): React.ReactNode {
    const { classes } = this.props

    let canvas = (
      <canvas
        key={`tag3d-canvas-${this.props.id}`}
        className={classes.tag3d_canvas}
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

    return canvas
  }

  /**
   * Handles canvas redraw
   *
   * @return {boolean}
   */
  public redraw(): boolean {
    if (this.canvas !== null && this._context !== null) {
      const labels = Session.label3dList.labels()
      this._context.clearRect(0, 0, this.canvas.width, this.canvas.height)
      for (const label of labels) {
        const category =
          label.category.length >= 1 &&
          label.category[0] < this._config.categories.length &&
          label.category[0] >= 0
            ? this._config.categories[label.category[0]]
            : ""
        const attributes = label.attributes
        const words = category.split(" ")
        let tag = words[words.length - 1]

        for (const attributeId of Object.keys(attributes)) {
          const attribute = this._config.attributes[Number(attributeId)]
          if (attribute.toolType === "switch") {
            if (attributes[Number(attributeId)][0] > 0) {
              tag += "," + attribute.tagText
            }
          } else if (attribute.toolType === "list") {
            if (attributes[Number(attributeId)][0] > 0) {
              tag +=
                "," +
                attribute.tagText +
                ":" +
                attribute.tagSuffixes[attributes[Number(attributeId)][0]]
            }
          }
        }

        const location = new THREE.Vector3().copy(label.center)
        location.project(this.camera)
        if (location.z > 0 && location.z < 1) {
          const x = ((location.x + 1) * this.canvas.width) / 2
          const y = ((-location.y + 1) * this.canvas.height) / 2
          this._context.font = "12px Verdana"
          this._context.fillStyle = "#FFFFFF"
          this._context.fillText(tag, x, y)
        }
      }
    }
    return true
  }

  /**
   * notify state is updated
   *
   * @param state
   */
  protected updateState(state: State): void {
    if (this.display !== this.props.display) {
      this._config = { ...state.task.config }
      this.display = this.props.display
      this.forceUpdate()
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
    const sensor = viewerConfig.sensor
    if (component === null || !isCurrentFrameLoaded(this.state, sensor)) {
      return
    }

    if (
      viewerConfig.type === ViewerConfigTypeName.IMAGE_3D ||
      viewerConfig.type === ViewerConfigTypeName.HOMOGRAPHY
    ) {
      this.data2d = true
    } else {
      this.data2d = false
    }

    if (component.nodeName === "CANVAS") {
      if (this.canvas !== component) {
        this.canvas = component
        this._context = this.canvas.getContext("2d")
        this.forceUpdate()
      }

      if (this.display !== null && this.data2d) {
        const img3dConfig = viewerConfig as Image3DViewerConfigType
        if (
          img3dConfig.viewScale >= MIN_SCALE &&
          img3dConfig.viewScale < MAX_SCALE
        ) {
          const newParams = updateCanvasScale(
            this.state,
            this.display,
            this.canvas,
            null,
            img3dConfig,
            img3dConfig.viewScale / this.scale,
            false
          )
          this.scale = newParams[3]
        }
      } else if (this.display !== null) {
        this.canvas.removeAttribute("style")
        const displayRect = this.display.getBoundingClientRect()
        this.canvas.width = displayRect.width
        this.canvas.height = displayRect.height
      }
    }
  }
}

const styledCanvas = withStyles(styles, { withTheme: true })(Tag3dCanvas)
export default connect(mapStateToDrawableProps)(styledCanvas)
