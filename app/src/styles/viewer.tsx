import { createStyles, StyleRules } from "@material-ui/core"

type ClassKey =
  | "viewer_container"
  | "viewer_button"
  | "camera_y_lock_icon"
  | "camera_x_lock_icon"

/**
 * Image viewer style
 */
export const viewerStyles = (): StyleRules<ClassKey, {}> =>
  createStyles({
    viewer_container: {
      display: "block",
      height: "100%",
      position: "absolute",
      overflow: "hidden",
      outline: "none",
      width: "100%",
      "touch-action": "none"
    },
    viewer_button: {
      color: "#ced4da",
      "z-index": 1001
    },
    camera_y_lock_icon: {
      "z-index": 1000
    },
    camera_x_lock_icon: {
      color: "#ced4da",
      transform: "rotate(90deg)",
      "z-index": 1000,
      "padding-top": "5px"
    }
  })
