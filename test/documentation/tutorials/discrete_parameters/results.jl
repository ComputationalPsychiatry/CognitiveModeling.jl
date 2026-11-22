#What the tutorial on discrete parameters gave when it ran: the variables of the model with the regression, which it
#reads last, split by kind, and a chain of the samples that Gibbs took with a sampler for each kind. The tutorial is a
#sketch which takes very few samples, so its fits aren't checked

@testset "The values of the sessions are discrete, and the coefficients continuous" begin
    @test discrete == [@varname(uses_rewards.sessions)]
    @test Set(continuous) ==
        Set([@varname(learning_rate.coef), @varname(uses_rewards.coef)])
end

@testset "Gibbs gives the samples that the tutorial asks for" begin
    @test size(chain) == (20, 1)
end
