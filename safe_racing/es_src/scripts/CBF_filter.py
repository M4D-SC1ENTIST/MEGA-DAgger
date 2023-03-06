agentsThreshold = 0.65 # safe minimum distance between agents' centers
borderThreshold = 0.42 # safe maximum distance between agent and boundary
eta_agents = 0.9 # parameter for discrete CBF, it could vary from 0 to 1
eta_border = 0.9

def CBFagent(x_ego, y_ego, x_opp, y_opp, x_ego_next, y_ego_next, x_opp_next, y_opp_next):

    """
        Check collisions between agents using CBF

        Args:
            x_ego: ego vehicle position x
            y_ego: ego vehicle position y
            x_opp: opponent vehicle position x
            y_opp: opponent vehicle position y
            x_ego_next: ego vehicle next position x
            y_ego_next: ego vehicle next position y
            x_opp_next: opponent vehicle next position x
            y_opp_next: opponent vehicle next position y

        Returns:
            True: safe
            False: unsafe

        TODO: write a more precise collision-checking CBF
    """

    currentCBFValue = (x_ego - x_opp) ** 2 + (y_ego - y_opp) ** 2 - agentsThreshold ** 2
    nextCBFValue = (x_ego_next - x_opp_next) ** 2 + (y_ego_next - y_opp_next) ** 2 - agentsThreshold ** 2
    CBFvalue = nextCBFValue - currentCBFValue + eta_agents * currentCBFValue
    #print("CBF value: ", CBFvalue)
    if CBFvalue >= - 0.1: # account for discretization error
        return True, CBFvalue
    else:
        return False, CBFvalue

def CBFborder(x_ego, y_ego, x_line, y_line, x_ego_next, y_ego_next, x_line_next, y_line_next):
    
    """
        Check collisions between agent and boundary using CBF

        Args:
            x_ego: ego vehicle position x
            y_ego: ego vehicle position y
            x_line: nearest boundary (one-side) position x
            y_line: nearest boundary (one-side) vehicle position y
            x_ego_next: ego vehicle next position x
            y_ego_next: ego vehicle next position y
            x_line_next: nearest boundary (one-side) position x
            y_line_next: nearest boundary (one-side) position y

        Returns:
            True: safe
            False: unsafe

        TODO: write a more precise collision-checking CBF
    """
    
    currentCBFValue = (x_ego - x_line) ** 2 + (y_ego - y_line) ** 2 - borderThreshold ** 2
    nextCBFValue = (x_ego_next - x_line_next) ** 2 + (y_ego_next - y_line_next) ** 2 - borderThreshold ** 2
    CBFvalue = nextCBFValue - currentCBFValue + eta_border * currentCBFValue
    print("CBF value: ", CBFvalue)
    if CBFvalue >= - 0.1: # account for discretization error
        return True, CBFvalue
    else:
        return False, CBFvalue
