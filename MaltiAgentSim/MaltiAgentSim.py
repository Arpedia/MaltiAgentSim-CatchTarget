# -*- encoding: utf-8 
from service.gameservice import GameService
from time import sleep
import numpy
import cv2
from matplotlib import pyplot

if __name__ == '__main__':

    # init
    h = 24
    w = 24
    area_shape = (h, w)
    mergin = 2
    epsilon = 0.3
    alpha = 0.3
    gamma = 0.7
    image_shape = (200, 200)

    # count
    total_plays = 500
    total_steps = numpy.zeros(total_plays).astype('int')
    play = 0
    steps = 0

    # construct
    gameservice = GameService()
    area, agent1, agent2, target = gameservice.construct(area_shape, mergin)

    # video writer
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    #writer = cv2.VideoWriter('q-learning.mp4', fourcc, 20, image_shape)
    
    while True:
        # disp
        print('play:' + str(play) + ' steps:'+ str(steps))

        # act
        gameservice.turn(area, agent1, agent2, target, epsilon)

        # update area and agents' q talbe
        agent1.update_q(area, alpha, gamma)
        agent2.update_q(area, alpha, gamma)

        # show image
        #image = cv2.resize(area.state[mergin:h - mergin, mergin:w - mergin], image_shape, interpolation = cv2.INTER_NEAREST)
        #writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        #cv2.imshow('', image)
        #if cv2.waitKey() == 27:
         #   break
        #sleep(0.1)

        # refresh state if agents catch the taget
        steps += 1
        if agent1.reward == 3:
            print('!!!catch the target!!!')
            gameservice.reset(area, agent1, agent2, target)
            agent1.save_q('q1.npy')
            agent2.save_q('q2.npy')
            total_steps[play] = steps
            steps = 0
            play += 1
            sleep(1)

        # count
        if play == total_plays:
            break

    pyplot.plot(numpy.arange(0, total_plays), total_steps)
    pyplot.savefig('graph.png')
    #cv2.destroyAllWindows()
    print('!!!finish!!!')