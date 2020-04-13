# agent environments
ENV_REACHER_SINGLE_AGENT    = 'env/Reacher_SingleAgent_Linux/Reacher.x86_64'
ENV_REACHER_MULTI_AGENTS    = 'env/Reacher_MultiAgent_Linux/Reacher.x86_64'
ENV_CRAWLER                 = 'env/Crawler_Linux/Crawler.x86_64'
ENV_TENNIS                  = 'env/Tennis_Linux/Tennis.x86_64'

# define environment dictionary
# set pretrained_weights to None if not using pretrained weights
env_dict = [{   
                'env':ENV_REACHER_SINGLE_AGENT, 
                'prefix':'reacher_single_agent', 
                'solved_score_thres':30.0,
                'pretrained_weights_actor':'weights/reacher_single_agent/checkpoint_actor.pth',
                'pretrained_weights_critic':'weights/reacher_single_agent/checkpoint_critic.pth'
            },
            {   
                'env':ENV_REACHER_MULTI_AGENTS, 
                'prefix':'reacher_multiple_agents', 
                'solved_score_thres':30.0,
                'pretrained_weights_actor':'weights/reacher_multiple_agents/checkpoint_actor.pth',
                'pretrained_weights_critic':'weights/reacher_multiple_agents/checkpoint_critic.pth'
            },
            {   
                'env':ENV_CRAWLER, 
                'prefix':'crawler', 
                'solved_score_thres':117.0,
                'pretrained_weights_actor':'weights/crawler/checkpoint_actor.pth',
                'pretrained_weights_critic':'weights/crawler/checkpoint_critic.pth'
            },
            {   
                'env':ENV_TENNIS, 
                'prefix':'tennis', 
                'solved_score_thres':0.5,
                'pretrained_weights_actor':'weights/tennis/checkpoint_actor.pth',
                'pretrained_weights_critic':'weights/tennis/checkpoint_critic.pth'
            }]

# env indexes
IDX_ENV_REACHER_SINGLE_AGENT    = 0
IDX_ENV_REACHER_MULTI_AGENTS    = 1
IDX_ENV_CRAWLER                 = 2
IDX_ENV_TENNIS                  = 3
