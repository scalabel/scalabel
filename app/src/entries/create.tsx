import { ThemeProvider } from "@material-ui/core/styles"
import React from "react"
import ReactDOM from "react-dom"

import Create from "../components/create_project"
import { scalabelTheme } from "../styles/theme"

ReactDOM.render(
  <ThemeProvider theme={scalabelTheme}>
    <Create />
  </ThemeProvider>,
  document.getElementById("create-root")
)
