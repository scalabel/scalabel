import List from "@material-ui/core/List"
import ListItem from "@material-ui/core/ListItem"
import ListItemText from "@material-ui/core/ListItemText"
import { withStyles } from "@material-ui/core/styles"
import Typography from "@material-ui/core/Typography"
import React, { ReactNode } from "react"

import { createStyle, formStyle } from "../styles/create"
import CreateForm from "./create_form"
import DividedPage from "./divided_page"
import ProjectList from "./sidebar_projects"

export interface ClassType {
  /** container of the list */
  listRoot: string
  /** list header (existing projects) */
  listHeader: string
}

export interface Props {
  /** Create classes */
  classes: ClassType
}

export interface State {
  /** boolean to force reload of the sidebar project list */
  reloadProjects: boolean
}

/**
 * Component which display the create page
 *
 * @param {object} props
 * @return component
 */
class Create extends React.Component<Props, State> {
  /**
   * create page header content
   */
  private readonly headerContent: ReactNode
  /**
   * create page sidebar content
   */
  private readonly sidebarContent: ReactNode
  /**
   * create page main content
   */
  private readonly mainContent: ReactNode

  /**
   * Constructor
   *
   * @param props
   */
  constructor(props: Props) {
    super(props)
    const { classes } = props
    this.state = {
      reloadProjects: false
    }
    this.headerContent = (
      <Typography variant="h6" noWrap>
        Open / Create Project
      </Typography>
    )
    this.sidebarContent = (
      <List className={classes.listRoot}>
        <ListItem>
          <ListItemText
            primary={
              <Typography className={classes.listHeader}>
                Existing Projects
              </Typography>
            }
            className={classes.listHeader}
          ></ListItemText>
        </ListItem>
        <ProjectList refresh={this.state.reloadProjects} />
      </List>
    )
    this.mainContent = (
      <StyledForm projectReloadCallback={this.projectReloadCallback} />
    )
  }

  /**
   * renders the create page
   *
   * @return component
   */
  public render(): React.ReactNode {
    return (
      <DividedPage
        header={this.headerContent}
        sidebar={this.sidebarContent}
        main={this.mainContent}
      />
    )
  }

  /**
   * callback used to force a state change to reload the project
   * list
   */
  private readonly projectReloadCallback = (): void => {
    this.setState({ reloadProjects: !this.state.reloadProjects })
  }
}

const StyledForm = withStyles(formStyle)(CreateForm)
/** export Create page */
export default withStyles(createStyle, { withTheme: true })(Create)
