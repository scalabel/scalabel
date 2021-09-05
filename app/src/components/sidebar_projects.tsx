import { Grid, Link } from "@material-ui/core"
import ListItem from "@material-ui/core/ListItem"
import CloseIcon from "@material-ui/icons/Close"
import { withStyles } from "@material-ui/core/styles"
import React from "react"

import { getAuth, getProjects, toProject } from "../common/service"
import { projectListStyle } from "../styles/create"
import { Endpoint } from "../const/connection"

interface ClassType {
  /** style for a colored entry */
  coloredListItem: string
}

// Props for project list
interface ProjectListProps {
  /** Project List classes */
  classes: ClassType
  /** refresh boolean */
  refresh: boolean
}

interface ProjectListState {
  /** boolean which when changed forces a refresh */
  reloadProjects: boolean
  /** projects */
  projectsToExpress: string[]
}

/** Project list sidebar component. Re-renders after
 * submission
 *
 * @param props
 */
class ProjectList extends React.Component<ProjectListProps, ProjectListState> {
  /** receive data from backend */
  // private projectsToExpress = getProjects()
  /**
   * Constructor
   *
   * @param props
   */
  public constructor(props: ProjectListProps) {
    super(props)
    this.state = {
      reloadProjects: props.refresh,
      projectsToExpress: getProjects()
    }
  }

  /**
   * method to process changes in props, which gets project from backend
   * and then changes state to force a reload of this component
   *
   * @param props
   */
  // public UNSAFE_componentWillReceiveProps(props: ProjectListProps): void {
  //   const { refresh } = this.props
  //   if (props.refresh !== refresh) {
  //     this.projectsToExpress = getProjects()
  //     this.setState({ reloadProjects: !this.state.reloadProjects })
  //   }
  // }

  /**
   * renders project list
   */
  public render(): React.ReactNode {
    const { classes } = this.props
    return (
      <div>
        {this.state.projectsToExpress.map((project, index) => (
          <ListItem
            button
            key={project}
            alignItems="center"
            className={index % 2 === 0 ? classes.coloredListItem : ""}
          >
            <Grid container justify="center">
              <Link
                component="button"
                variant="body2"
                color="inherit"
                onClick={() => {
                  toProject(project)
                }}
              >
                {project}
              </Link>
            </Grid>
            {this.state.projectsToExpress[0] !== "No existing project" ? (
              <CloseIcon
                titleAccess="Delete project"
                fontSize="small"
                onClick={() => this.deleteProject(project)}
              />
            ) : null}
          </ListItem>
        ))}
      </div>
    )
  }

  /**
   * This function will delete project according to the project name
   *
   * @param projectName
   */
  public deleteProject(projectName: string): void {
    const xhr = new XMLHttpRequest()
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        this.setState({ projectsToExpress: getProjects() })
      }
    }
    xhr.open("GET", `${Endpoint.DELETE_PROJECT}?project_name=` + projectName)
    const auth = getAuth()
    if (auth !== "") {
      xhr.setRequestHeader("Authorization", auth)
    }
    xhr.send()
  }
}

export default withStyles(projectListStyle)(ProjectList)
