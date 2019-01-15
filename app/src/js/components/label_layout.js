import React from 'react';
import PropTypes from 'prop-types';
import CssBaseline from '@material-ui/core/CssBaseline';
import SplitPane from 'react-split-pane';
import Session from '../common/session';

type Props = {
  titleBar: Object,
  leftSidebar1: Object,
  leftSidebar2?: Object,
  main: Object,
  bottomBar?: Object,
  rightSidebar1?: Object,
  rightSidebar2?: Object,
}

type State = {
  left_size: number,
  center_size: number,
  right_size: number,
}

window.__MUI_USE_NEXT_TYPOGRAPHY_VARIANTS__ = true;

/**
 * Layout of the labeling interface
 */
class LabelLayout extends React.Component<Props, State> {
  /**
   * @param {object} props
   */
  constructor(props: Object) {
    super(props);
    this.state = {left_size: 0, center_size: 0, right_size: 0};
    Session.subscribe(this);
  }

  /**
   * called on redux store update
   */
  onStateUpdated() {
    this.setState(this.state);
  }

  /**
   * @param {number} size
   * @param {string} position
   */
  handleOnChange(size: number, position: string) {
    let state = this.state;
    if (position === 'left' && this.state.left_size !== size) {
      state.left_size = size;
    } else if (position === 'center' && this.state.center_size !== size) {
      state.center_size = size;
    } else if (position === 'right' && this.state.right_size !== size) {
      state.right_size = size;
    }
    this.setState(state);
  }

  /**
   * Split component with the second component optional
   * @param {string} split - horizontal or vertical
   * @param {React.Fragment} comp1 - the first component
   * @param {React.Fragment} comp2 - the second component
   * @param {string} name1 - the class name of the first component
   * @param {string} name2 - the class name of the second component
   * @param {number} min - the minimum size
   * @param {number} dflt - the default size
   * @param {number} max - the maximum size
   * @param {string} primary - which component the size constraint is for
   * the second component
   * @param {string} position - left, center or right:
   * which size to update in state
   * @return {Component}
   */
  optionalSplit(split: string, comp1: React.Fragment, comp2: React.Fragment,
                name1: string, name2: string, min: number, dflt: number,
                max: number, primary: string = 'first',
                position: string = 'center') {
    if (!comp1) {
      return;
    }
    return (
        comp2 ?
            <SplitPane split={split} minSize={min}
                       defaultSize={dflt}
                       maxSize={max} primary={primary}
                       onChange={(size) => {
                         this.handleOnChange(size, position);
                       }}>
              <div className={name1}>
                {comp1}
              </div>
              <div className={name2}>
                {comp2}
              </div>
            </SplitPane>
            : <div className={name1}>
              {comp1}
            </div>
    );
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  render() {
    const {titleBar, leftSidebar1, leftSidebar2, bottomBar,
      main, rightSidebar1, rightSidebar2} = this.props;
    const mainWithProps = React.cloneElement(main, {});

    const leftDefaultWidth = 200;
    const leftMaxWidth = 300;
    const leftMinWidth = 180;
    const rightDefaultWidth = 200;
    const rightMaxWidth = 300;
    const rightMinWidth = 180;
    const topDefaultHeight = 200;
    const topMaxHeight = 300;
    const topMinHeight = 180;
    const bottomDefaultHeight = 200;
    const bottomMaxHeight = 300;
    const bottomMinHeight = 180;

    return (
        <React.Fragment>
          <CssBaseline />
          <div className='titleBar'>
            {titleBar}
          </div>
          <main>
            {this.optionalSplit('vertical',

              // left sidebar
              this.optionalSplit('horizontal', leftSidebar1, leftSidebar2,
                'leftSidebar1', 'leftSidebar2',
                topMinHeight, topDefaultHeight, topMaxHeight,
                'first'),

              this.optionalSplit('vertical',

                // center
                this.optionalSplit('horizontal', mainWithProps, bottomBar,
                'main', 'bottomBar',
                bottomMinHeight, bottomDefaultHeight, bottomMaxHeight,
                'second', 'center'),

                // right sidebar
                this.optionalSplit('horizontal', rightSidebar1, rightSidebar2,
                'rightSidebar1', 'rightSidebar2',
                topMinHeight, topDefaultHeight, topMaxHeight),
                'center', 'rightSidebar',
                rightMinWidth, rightDefaultWidth, rightMaxWidth, 'second',
                'right'
                ),

                'leftSidebar', 'centerAndRightSidebar',
                leftMinWidth, leftDefaultWidth, leftMaxWidth, 'first', 'left'
              )}
          </main>
          {/* End footer */}
        </React.Fragment>
    );
  }
}

LabelLayout.propTypes = {
  titleBar: PropTypes.object.isRequired,
  leftSidebar1: PropTypes.object.isRequired,
  leftSidebar2: PropTypes.object,
  main: PropTypes.object.isRequired,
  bottomBar: PropTypes.object,
  rightSidebar1: PropTypes.object,
  rightSidebar2: PropTypes.object,
};

export default LabelLayout;
