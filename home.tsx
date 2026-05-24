import { Link } from "wouter";
import { useGetDiseaseStats, useListPlants } from "@workspace/api-client-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowRight, Leaf, ShieldAlert, Activity, Microscope } from "lucide-react";

export default function Home() {
  const { data: stats, isLoading: isStatsLoading } = useGetDiseaseStats();
  const { data: plants, isLoading: isPlantsLoading } = useListPlants();

  return (
    <div className="w-full">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-primary/5 pt-16 md:pt-24 lg:pt-32 pb-16 md:pb-20 lg:pb-28">
        <div className="absolute inset-0 z-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: "url('/images/hero.png')", backgroundSize: 'cover', backgroundPosition: 'center', filter: 'grayscale(100%) contrast(150%)' }} />
        <div className="container relative z-10 mx-auto px-4 md:px-6">
          <div className="grid gap-12 lg:grid-cols-2 lg:gap-8 items-center">
            <div className="flex flex-col justify-center space-y-8">
              <div className="space-y-4">
                <div className="inline-flex items-center rounded-full border bg-background/50 px-3 py-1 text-sm text-muted-foreground backdrop-blur-sm shadow-sm">
                  <span className="flex h-2 w-2 rounded-full bg-primary mr-2 animate-pulse"></span>
                  ML-Powered Plant Diagnostics
                </div>
                <h1 className="font-serif text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl text-foreground">
                  Protect your crop with <span className="text-primary italic">scientific precision</span>
                </h1>
                <p className="max-w-[600px] text-lg text-muted-foreground md:text-xl leading-relaxed">
                  LeafSentry combines expert botanical research with machine learning to identify, understand, and treat plant diseases before they spread.
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-4">
                <Link href="/health-check">
                  <Button size="lg" className="w-full sm:w-auto text-base h-12 px-8 shadow-md">
                    <Microscope className="mr-2 h-5 w-5" />
                    Run Health Check
                  </Button>
                </Link>
                <Link href="/diseases">
                  <Button variant="outline" size="lg" className="w-full sm:w-auto text-base h-12 px-8 bg-background/50 backdrop-blur-sm">
                    Browse Library
                  </Button>
                </Link>
              </div>
              
              <div className="pt-4 flex items-center space-x-8 text-sm text-muted-foreground">
                <div className="flex items-center">
                  <ShieldAlert className="mr-2 h-4 w-4 text-primary" />
                  <span>Trusted by Agronomists</span>
                </div>
                <div className="flex items-center">
                  <Activity className="mr-2 h-4 w-4 text-primary" />
                  <span>Real-time Analysis</span>
                </div>
              </div>
            </div>
            
            <div className="mx-auto w-full max-w-[500px] lg:max-w-none relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-primary to-accent rounded-2xl blur opacity-20 animate-pulse" />
              <div className="relative aspect-[4/3] sm:aspect-[16/9] lg:aspect-square overflow-hidden rounded-2xl border shadow-xl bg-muted/20">
                <img
                  src="https://images.unsplash.com/photo-1566750948564-cd7f3a4f8c98?q=80&w=800&auto=format&fit=crop"
                  alt="Modern botanical research greenhouse"
                  className="object-cover w-full h-full"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="border-y bg-background py-12">
        <div className="container mx-auto px-4 md:px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 divide-x divide-border/50">
            {[
              { label: "Plant Species", value: stats?.totalPlants, icon: Leaf },
              { label: "Diseases Tracked", value: stats?.totalDiseases, icon: ShieldAlert },
              { label: "Assessments Run", value: stats?.assessmentsRun, icon: Activity },
              { label: "Critical Threats", value: stats?.criticalDiseases, icon: Microscope },
            ].map((stat, i) => (
              <div key={i} className="flex flex-col items-center justify-center text-center px-4 first:border-l-0">
                {isStatsLoading ? (
                  <>
                    <Skeleton className="h-10 w-16 mb-2" />
                    <Skeleton className="h-4 w-24" />
                  </>
                ) : (
                  <>
                    <div className="text-3xl md:text-4xl font-bold tracking-tighter text-foreground mb-1 font-serif">
                      {stat.value?.toLocaleString() || "0"}
                    </div>
                    <div className="text-sm font-medium text-muted-foreground flex items-center justify-center">
                      <stat.icon className="h-3.5 w-3.5 mr-1.5 opacity-70" />
                      {stat.label}
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Featured Plants */}
      <section className="py-20 md:py-28 bg-muted/10">
        <div className="container mx-auto px-4 md:px-6">
          <div className="flex flex-col md:flex-row justify-between items-end mb-12 gap-4">
            <div className="max-w-2xl">
              <h2 className="font-serif text-3xl font-bold tracking-tight sm:text-4xl mb-4">Supported Species</h2>
              <p className="text-muted-foreground text-lg">
                Our models and research currently focus on these key agricultural species, providing comprehensive disease profiles and treatment protocols.
              </p>
            </div>
            <Link href="/plants">
              <Button variant="ghost" className="group">
                View all plants
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </div>

          {isPlantsLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
              {[1, 2, 3].map(i => (
                <Card key={i} className="overflow-hidden">
                  <Skeleton className="h-48 w-full rounded-none" />
                  <CardContent className="p-6">
                    <Skeleton className="h-6 w-1/2 mb-3" />
                    <Skeleton className="h-4 w-1/3 mb-4" />
                    <Skeleton className="h-20 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
              {plants?.slice(0, 3).map((plant) => (
                <Link key={plant.id} href={`/plants/${plant.id}`} className="block group">
                  <Card className="h-full overflow-hidden transition-all duration-300 hover:shadow-lg hover:-translate-y-1 border-border/50 bg-background/50 hover:bg-background">
                    <div className="aspect-[4/3] overflow-hidden bg-muted relative">
                      {plant.imageUrl ? (
                        <img 
                          src={plant.imageUrl} 
                          alt={plant.name} 
                          className="object-cover w-full h-full transition-transform duration-500 group-hover:scale-105" 
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center bg-primary/10">
                          <Leaf className="h-12 w-12 text-primary/30" />
                        </div>
                      )}
                      <div className="absolute top-4 right-4 bg-background/90 backdrop-blur text-foreground text-xs font-semibold px-2.5 py-1 rounded-full shadow-sm">
                        {plant.diseaseCount} Diseases
                      </div>
                    </div>
                    <CardContent className="p-6">
                      <h3 className="font-serif text-xl font-bold mb-1 group-hover:text-primary transition-colors">{plant.name}</h3>
                      <p className="text-sm font-medium text-muted-foreground italic mb-4">{plant.scientificName}</p>
                      <p className="text-sm text-muted-foreground line-clamp-3 leading-relaxed">
                        {plant.description}
                      </p>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 relative overflow-hidden bg-primary text-primary-foreground">
        <div className="absolute inset-0 z-0 opacity-10 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-white to-transparent" />
        <div className="container relative z-10 mx-auto px-4 md:px-6 text-center max-w-3xl">
          <Leaf className="h-12 w-12 mx-auto mb-6 opacity-80" />
          <h2 className="font-serif text-3xl font-bold tracking-tight sm:text-4xl mb-6">
            Identify diseases instantly
          </h2>
          <p className="text-primary-foreground/80 text-lg md:text-xl mb-10 leading-relaxed">
            Upload a photo of a symptomatic leaf and our ML model will analyze it against our database of plant diseases, providing immediate confidence scores and treatment recommendations.
          </p>
          <Link href="/health-check">
            <Button size="lg" variant="secondary" className="h-14 px-8 text-base font-semibold shadow-xl hover:scale-105 transition-transform">
              Start Assessment
            </Button>
          </Link>
        </div>
      </section>
    </div>
  );
}
