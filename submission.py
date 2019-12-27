from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum


def dist_manhattan(pos1 : tuple, pos2 : tuple)-> int:
    return abs(pos1[0]-pos2[0])+abs(pos1[1]-pos2[1])


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    # Insert your code here...

    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length

    # calculate the sum of the dist of snake head from the two closest walls in board (x and y axles)
    head_position = state.snakes[player_index].head
    distance_from_closest_walls = min(state.board_size[0]-head_position[0],head_position[0]) \
                                  + min(state.board_size[1]-head_position[1],head_position[1])

    distance_from_closest_fruit = min([dist_manhattan(fruit, head_position) for fruit in state.fruits_locations])

    closest_rival_snake = state.board_size[0]+state.board_size[1]

    for rival in range(state.n_agents):
        tmp_min = state.board_size[0] + state.board_size[1]
        if state.snakes[rival].alive and rival != player_index:
            tmp = (tmp_min, tmp_min)
            for pos in state.snakes[rival].position:
                if dist_manhattan(pos, state.snakes[player_index].head) < tmp_min:
                    tmp_min = dist_manhattan(pos, state.snakes[player_index].head)
                    tmp = (tmp_min, state.snakes[rival].position.index(pos))
    # tmp(0) - dist from head of the snake to the closest part of the rival
    # tmp(1) - difference from closest part of rival from its tail

            if tmp[0] < tmp[1] and tmp[0] < closest_rival_snake:
                closest_rival_snake = tmp[0]

    if closest_rival_snake is state.board_size[0] + state.board_size[1]:
        closest_rival_snake = 0

    w_of_walls = 0.5
    w_of_fruits = 2
    w_of_rival = 2
    w_of_len = 30

    return w_of_len * state.snakes[player_index].length + w_of_walls*distance_from_closest_walls + \
           w_of_rival*closest_rival_snake - w_of_fruits*distance_from_closest_fruit


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """
    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """
        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    # aux functions

    def RB_minimax(self, state: TurnBasedGameState, depth, isAlphaBeta=False, alpha=-np.inf, beta=np.inf):
        # stop conditions - if we reached a final state (#turns or snake dead) or reached final depth
        if depth == 0:
            return heuristic(state.game_state, self.player_index), state.agent_action

        if state.game_state.is_terminal_state:
            if state.game_state.current_winner.player_index == self.player_index:
                return np.inf, state.agent_action
            else:
                return -np.inf, state.agent_action

        # players turn
        if state.turn == self.Turn.AGENT_TURN:
            curr_max = -np.inf
            curr_move = (curr_max, GameAction.LEFT)
            for succ in state.game_state.get_possible_actions():
                assert succ is not None
                next_state = self.TurnBasedGameState(state.game_state, succ)
                tmp_sum, tmp_move = self.RB_minimax(next_state, depth, isAlphaBeta, alpha, beta)

                if curr_max < tmp_sum:
                    curr_max = tmp_sum
                    curr_move = (tmp_sum, succ)

                # curr_max = max(tmp, curr_max)

                if isAlphaBeta:                     # check if calculates an alpha-beta algorithm
                    alpha = max(alpha, curr_max)
                    if curr_max >= beta:
                        return np.inf, succ    # curr_move in this case is irrelevant
            return curr_move

        # opponents turn
        else:
            curr_min = np.inf
            curr_move = (curr_min, GameAction.LEFT)
            # iterate over all possible actions of opponents
            for succ in state.game_state.get_possible_actions_dicts_given_action(state.agent_action, self.player_index):
                succ[self.player_index] = state.agent_action
                next_state = self.TurnBasedGameState(get_next_state(state.game_state, succ), None)
                tmp_sum, tmp_move = self.RB_minimax(next_state, depth-1, isAlphaBeta, alpha, beta)

                if curr_min > tmp_sum:
                    curr_min = tmp_sum
                    curr_move = (tmp_sum, state.agent_action)

                # curr_min = min(tmp, curr_min)

                if isAlphaBeta:                             # check if calculates an alpha-beta algorithm
                    beta = min(curr_min, beta)
                    if curr_min <= alpha:
                        return -np.inf, state.agent_action  # curr_move in this case is irrelevant
            return curr_move

    def get_action(self, state: GameState) -> GameAction:
        # check all 3 possible moves of the player
        '''
        left = MinimaxAgent.TurnBasedGameState(state, GameAction.LEFT)
        res_left = self.RB_minimax(left,2)
        straight = MinimaxAgent.TurnBasedGameState(state, GameAction.STRAIGHT)
        res_straight = self.RB_minimax(straight, 2)
        right = MinimaxAgent.TurnBasedGameState(state, GameAction.RIGHT)
        res_right = self.RB_minimax(right, 2)

        # return the best move (highest minimax result) out if the three
        if res_left > res_right and res_left > res_straight:
            return GameAction.LEFT
        elif res_straight > res_right:
            return GameAction.STRAIGHT
        else:
            return GameAction.RIGHT

            '''
        depth = 2
        start_state = MinimaxAgent.TurnBasedGameState(state, None)
        result_sum, result_action = self.RB_minimax(start_state, depth)
        assert result_action is not None
        return result_action


class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        depth = 2
        start_state = MinimaxAgent.TurnBasedGameState(state, None)
        result_sum, result_action = self.RB_minimax(start_state, depth, True)
        if type(result_action) is not GameAction:
            print(type(result_action))
        return result_action




        pass


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    pass


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()

